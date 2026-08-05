#!/usr/bin/env python3
# scripts/compare_viz.py
"""
Compare two visualization files (VTK family) field-by-field with tolerances.

Usage:
  # direct PVTU, VTU files
  python3 scripts/compare_viz.py out/ParaView/step_00000.vtu out/ParaView/step_00100.vtu \
      --fields rho,u,v,p --rtol 1e-10 --atol 1e-12

  # direct PVD (compares data for each/all timesteps!)
  python3 scripts/compare_viz.py case1_out/ParaView/case.pvd case2_out/ParaView/case.pvd \
      --fields rho,u,v,p --rtol 1e-10 --atol 1e-12

  # PVD convenience (e.g. for cyclic cases): pass a single .pvd and it will compare first vs last dataset
  python3 scripts/compare_viz.py out/ParaView/case.pvd --fields rho,u,v,p

Exit codes:
  0 -> pass, 1 -> fail, 2 -> input error
"""
from __future__ import annotations
import argparse
import sys
import os
import json
import hashlib
from typing import Dict, List, Tuple
import numpy as np

try:
    import xml.etree.ElementTree as ET
except Exception:
    ET = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two VTK viz files with tolerances.")
    p.add_argument("inputs", nargs="+",
                   help="Two viz files of type pvtu, pvd, or vtu. OR a single PVD to compare first vs. last.")
    p.add_argument("--fields", default="", help="Comma-separated list of fields to compare (empty = all common).")
    p.add_argument("--exclude-fields", default="", help="Comma-separated list of fields to ignore.")
    p.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance.")
    p.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance.")
    p.add_argument("--compare-cells", action="store_true",
                   help="Also compare cell_data (default compares only point_data).")
    p.add_argument("--json-out", default="", help="Optional path to write machine-readable diff JSON.")
    p.add_argument("--quiet", action="store_true", help="Less verbose output.")
    return p.parse_args()


def _hash(a: np.ndarray) -> str:
    # Stable content hash for quick identity checks (after canonicalization)
    h = hashlib.blake2b(digest_size=12)
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def _canonicalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted indices for points by (x,y,z) and the sorted array."""
    if points.ndim != 2:
        raise ValueError("points must be (N,dim)")
    key = np.lexsort(points[:, :3].T[::-1])  # sort by x, then y, then z
    return key, points[key]


def _cell_centers(points: np.ndarray, cells: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    """Compute cell centers as mean of vertices; return array aligned to concatenated cell order."""
    centers = []
    for _cell_type, conn in cells:
        if conn.size == 0:
            continue
        pts = points[conn]
        # mean over node index axis
        c = pts.mean(axis=1)
        centers.append(c)
    if not centers:
        return np.zeros((0, points.shape[1]))
    return np.vstack(centers)


def _canonicalize_cells(mesh, prefer_backend_centers: bool = False):
    """
    Return (sort_index, sorted_centers).
    For PyVista meshes, prefer mesh.cell_centers() to guarantee same order as cell_data.
    For meshio meshes, compute centers from connectivity blocks in given order.
    """
    # PyVista adapter?
    if hasattr(mesh, "_pv"):  # see adapter below
        pv = mesh._pv
        if prefer_backend_centers:
            centers = np.asarray(pv.cell_centers().points)
            if centers.size == 0:
                return np.array([], dtype=int), centers
            key = np.lexsort(centers[:, :3].T[::-1])
            return key, centers[key]

        # fallback: compute with connectivity (rarely needed if centers() is available)
        # (not shown; the centers() path is preferred)

    # meshio-like path
    centers = _cell_centers(np.asarray(mesh.points), mesh.cells)
    if centers.shape[0] == 0:
        return np.array([], dtype=int), centers
    key = np.lexsort(centers[:, :3].T[::-1])
    return key, centers[key]


def _stack_point_data(mesh) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    npts = int(np.asarray(mesh.points).shape[0])

    for name, arr in mesh.point_data.items():
        a = np.asarray(arr)

        # Skip non-numeric or obvious VTK internals
        if a.dtype.kind in {"U", "S", "O"} or name.startswith("vtk"):
            continue

        if a.ndim == 1:
            # Expect (Npts,) -> make (Npts, 1)
            if a.size == npts:
                a = a.reshape(npts, 1)
            else:
                # Not point-sized data; ignore
                continue
        elif a.ndim == 2:
            # Expect (Npts, comp). If (comp, Npts), transpose.
            if a.shape[0] == npts:
                pass
            elif a.shape[1] == npts:
                a = a.T
            else:
                # Mismatch; ignore
                continue
        else:
            # Flatten trailing dims if total size is a multiple of Npts
            if a.size % npts != 0:
                continue
            a = a.reshape(npts, -1)

        out[name] = np.ascontiguousarray(a)

    return out


def _canonicalize_legacy_fields(fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map legacy visualization arrays to their canonical representation."""
    canonical = dict(fields)
    legacy_velocity = ["Horizontal V", "Vertical V", "Normal V"]

    if "Velocity" not in canonical and "Horizontal V" in canonical:
        components = []
        for name in legacy_velocity:
            if name not in canonical:
                break
            component = canonical[name]
            if component.ndim != 2 or component.shape[1] != 1:
                raise ValueError(
                    f"legacy velocity field '{name}' must have exactly one component"
                )
            components.append(component)
        canonical["Velocity"] = np.hstack(components)

    return canonical


def _stack_cell_data(mesh) -> Dict[str, np.ndarray]:
    # PyVista adapter: already (Nc, comp) arrays wrapped in a one-element list
    if hasattr(mesh, "_pv"):
        return {n: v[0] if isinstance(v, list) and len(v) == 1 else np.asarray(v)
                for n, v in mesh.cell_data.items()}

    # meshio path: stack per block
    if not getattr(mesh, "cell_data", None):
        return {}
    out: Dict[str, List[np.ndarray]] = {}
    for name, blocks in mesh.cell_data.items():
        stk: List[np.ndarray] = []
        for blk in blocks:
            b = np.asarray(blk)
            if b.ndim == 1:
                b = b.reshape(-1, 1)
            elif b.ndim == 2 and b.shape[0] < b.shape[1]:
                b = b.T
            else:
                b = b.reshape(b.shape[0], -1)
            stk.append(np.ascontiguousarray(b))
        if stk:
            out[name] = np.vstack(stk)
    return out


def _select_common_fields(A: Dict[str, np.ndarray], B: Dict[str, np.ndarray],
                          include: List[str] | None, exclude: List[str]) -> List[str]:
    common = sorted(set(A.keys()) & set(B.keys()))
    if include:
        common = [f for f in common if f in include]
    if exclude:
        ex = set(exclude)
        common = [f for f in common if f not in ex]
    return common


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> Dict[str, float]:
    # a,b: (N, comp) aligned
    if a.shape != b.shape:
        return {"shape_mismatch": 1.0}
    diff = b - a
    abs_err = np.abs(diff)
    # norms per-component collapsed
    l2 = float(np.linalg.norm(diff) / max(1.0, np.linalg.norm(a)))
    linf = float(abs_err.max(initial=0.0))
    mean_abs = float(abs_err.mean())
    # pass/fail: use elementwise tolerance check
    ok = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return {"l2_rel": l2, "linf_abs": linf, "mean_abs": mean_abs, "ok": bool(ok)}


def _load_mesh(path: str) -> 'MeshAdapter':
    import pyvista as pv
    # pyvista wraps vtkXML{P,U}UnstructuredGridReader internally
    mesh = pv.read(path)

    # For parallel XML files (.pvtu, etc.) pyvista may return a MultiBlock.
    # Collapse to a single UnstructuredGrid so the rest of the script can
    # treat it like a normal mesh.
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()   # merges all blocks into one unstructured grid

    # Create a fake meshio-like object interface so the rest of the script works
    class MeshAdapter:
        def __init__(self, m):
            self._pv = m  # keep pyvista object for backend features
            self.points = np.array(m.points)
            # pyvista stores both point_data and cell_data
            self.point_data = {k: np.array(v) for k, v in m.point_data.items()}
            # store as a list of blocks to match the expectations of _stack_cell_data
            self.cell_data = {k: [np.array(v)] for k, v in m.cell_data.items()}
            self.cell_types = np.array(m.celltypes)
            self.cells = []
    return MeshAdapter(mesh)


def _from_pvd(pvd_path: str) -> Tuple[str, str]:
    """Return (first_dataset_path, last_dataset_path) from a .pvd file."""
    if ET is None:
        raise RuntimeError("xml parser unavailable; cannot read .pvd")
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    sets = root.findall(".//DataSet")
    if not sets:
        raise RuntimeError("No DataSet entries in PVD.")
    # preserve file order; some PVDs include timestep attribute but order is fine
    first = sets[0].attrib.get("file")
    last = sets[-1].attrib.get("file")
    base = os.path.dirname(pvd_path)
    return os.path.join(base, first), os.path.join(base, last)


def _list_from_pvd(pvd_path: str) -> List[str]:
    """Return a list of dataset paths from a .pvd file, in file order."""
    if ET is None:
        raise RuntimeError("xml parser unavailable; cannot read .pvd")
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    sets = root.findall(".//DataSet")
    if not sets:
        raise RuntimeError("No DataSet entries in PVD.")
    base = os.path.dirname(pvd_path)
    return [os.path.join(base, ds.attrib.get("file")) for ds in sets]


def compare_files(f0: str, f1: str, fields: List[str], exclude: List[str],
                  rtol: float, atol: float, compare_cells: bool, quiet: bool) -> Tuple[bool, Dict]:
    m0 = _load_mesh(f0)
    m1 = _load_mesh(f1)

    pts0 = np.asarray(m0.points)
    pts1 = np.asarray(m1.points)

    if pts0.shape == pts1.shape and np.allclose(pts0, pts1, rtol=0.0, atol=2e-15):
        idx0 = np.arange(pts0.shape[0])
        idx1 = np.arange(pts1.shape[0])
        pts0_sorted = pts0
        pts1_sorted = pts1
    else:
        # Canonicalize point ordering
        idx0, pts0_sorted = _canonicalize_points(m0.points)
        idx1, pts1_sorted = _canonicalize_points(m1.points)

    # Geometry must contain the same point locations, independent of ordering.
    bbox0 = np.array([pts0_sorted.min(axis=0), pts0_sorted.max(axis=0)])
    bbox1 = np.array([pts1_sorted.min(axis=0), pts1_sorted.max(axis=0)])
    geom_ok = (
        pts0_sorted.shape == pts1_sorted.shape
        and np.allclose(pts0_sorted, pts1_sorted, rtol=1e-12, atol=1e-12)
    )

    # A cell-type histogram catches representation changes such as one
    # high-order Lagrange cell versus multiple linear GLL subcells.
    cell_types0 = getattr(m0, "cell_types", None)
    cell_types1 = getattr(m1, "cell_types", None)
    topology_ok = True
    if cell_types0 is not None and cell_types1 is not None:
        types0, counts0 = np.unique(cell_types0, return_counts=True)
        types1, counts1 = np.unique(cell_types1, return_counts=True)
        topology_ok = np.array_equal(types0, types1) and np.array_equal(counts0, counts1)

    # Point data (aligned by sorted point index)
    pd0_raw = _canonicalize_legacy_fields(_stack_point_data(m0))
    pd1_raw = _canonicalize_legacy_fields(_stack_point_data(m1))

    pd0 = {k: v[idx0] for k, v in pd0_raw.items()}
    pd1 = {k: v[idx1] for k, v in pd1_raw.items()}

    # Cells: concatenate & canonicalize on centers
    cd0 = {}
    cd1 = {}
    if compare_cells:
        cd0_raw = _stack_cell_data(m0)
        cd1_raw = _stack_cell_data(m1)
        ckey0, _ = _canonicalize_cells(m0, prefer_backend_centers=True)
        ckey1, _ = _canonicalize_cells(m1, prefer_backend_centers=True)
        cd0 = {k: (v[ckey0] if v.size else v) for k, v in cd0_raw.items()}
        cd1 = {k: (v[ckey1] if v.size else v) for k, v in cd1_raw.items()}
        # ckey0, _ = _canonicalize_cells(m0.points, m0.cells)
        for side, key, cd in (("A", ckey0, cd0_raw), ("B", ckey1, cd1_raw)):
            for nm, arr in cd.items():
                if arr.size and key.size and arr.shape[0] != key.size:
                    raise RuntimeError(
                        f"Cell count mismatch on {side} for '{nm}': "
                        f"cell_data rows={arr.shape[0]} vs centers={key.size}. "
                        "This indicates a backend ordering mismatch."
                    )

    include = [f.strip() for f in fields if f.strip()]
    exclude = [f.strip() for f in exclude if f.strip()]

    available_fields = set(pd0) & set(pd1)
    if compare_cells:
        available_fields |= set(cd0) & set(cd1)
    missing_fields = sorted(set(include) - available_fields)
    if missing_fields:
        raise ValueError(
            "requested fields are not present in both files: "
            + ", ".join(missing_fields)
        )

    results: Dict = {
        "files": [f0, f1],
        "geom_ok": bool(geom_ok),
        "topology_ok": bool(topology_ok),
        "point_data": {},
        "cell_data": {},
    }

    # Which fields
    pfields = _select_common_fields(pd0, pd1, include or None, exclude)
    for name in pfields:
        r = _compare_arrays(pd0[name], pd1[name], rtol, atol)
        r["ncomp"] = int(pd0[name].shape[1])
        r["n"] = int(pd0[name].shape[0])
        r["hash0"] = _hash(pd0[name])
        r["hash1"] = _hash(pd1[name])
        results["point_data"][name] = r

    cfields = []
    if compare_cells:
        cfields = _select_common_fields(cd0, cd1, include or None, exclude)
        for name in cfields:
            r = _compare_arrays(cd0[name], cd1[name], rtol, atol)
            r["ncomp"] = int(cd0[name].shape[1])
            r["n"] = int(cd0[name].shape[0])
            r["hash0"] = _hash(cd0[name])
            r["hash1"] = _hash(cd1[name])
            results["cell_data"][name] = r

    if not pfields and not cfields:
        if compare_cells:
            raise ValueError("no common point- or cell-data fields to compare")
        raise ValueError("no common point-data fields to compare")

    # overall pass
    ok_fields = all(v.get("ok", False) for v in results["point_data"].values())
    if compare_cells:
        ok_fields = ok_fields and all(v.get("ok", False) for v in results["cell_data"].values())
    results["ok"] = bool(geom_ok and topology_ok and ok_fields)

    if not quiet:
        print(f"[compare_viz] Comparing:\n  A: {f0}\n  B: {f1}")
        print(f"  Geometry point match: {geom_ok} (A bbox:{bbox0.tolist()} B bbox:{bbox1.tolist()})")
        print(f"  Cell topology match: {topology_ok}")
        def _pp(section, data):
            if not data:
                print(f"  {section}: (no common fields)")
                return
            print(f"  {section}:")
            for k, r in data.items():
                if "shape_mismatch" in r:
                    print(f"    {k:20s} FAIL  shape mismatch")
                    continue
                status = "OK" if r["ok"] else "FAIL"
                print(f"    {k:20s} {status:4s}  n={r['n']:7d} comp={r['ncomp']}"
                      f"  L2rel={r['l2_rel']:.3e}  Linf={r['linf_abs']:.3e}  mean|e|={r['mean_abs']:.3e}"
                      f"  hashA={r['hash0'][:6]} hashB={r['hash1'][:6]}")
        _pp("point_data", results["point_data"])
        if compare_cells:
            _pp("cell_data", results["cell_data"])
        print(f"Overall: {'PASS' if results['ok'] else 'FAIL'}  (rtol={rtol}, atol={atol})")

    return results["ok"], results

def main():
    args = parse_args()
    inputs = args.inputs

    pairs: List[Tuple[str, str]] = []

    # Case 1: single .pvd -> compare first vs last inside it
    if len(inputs) == 1 and inputs[0].lower().endswith(".pvd"):
        pvd = inputs[0]
        try:
            f0, f1 = _from_pvd(pvd)
        except Exception as e:
            print(f"ERROR reading PVD: {e}", file=sys.stderr)
            return 2
        pairs.append((f0, f1))

    # Case 2: two .pvd files -> compare timestep-by-timestep
    elif (
        len(inputs) == 2
        and inputs[0].lower().endswith(".pvd")
        and inputs[1].lower().endswith(".pvd")
    ):
        pvd0, pvd1 = inputs
        try:
            seq0 = _list_from_pvd(pvd0)
            seq1 = _list_from_pvd(pvd1)
        except Exception as e:
            print(f"ERROR reading PVDs: {e}", file=sys.stderr)
            return 2

        if len(seq0) != len(seq1):
            print(
                f"ERROR: PVD files have different number of datasets "
                f"({len(seq0)} vs {len(seq1)}).",
                file=sys.stderr,
            )
            return 2

        pairs = list(zip(seq0, seq1))

    # Case 3: two direct files (vtu/pvtu/etc.)
    elif len(inputs) == 2:
        pairs.append((inputs[0], inputs[1]))

    else:
        print(
            "ERROR: Provide either two files, two .pvd files, or a single .pvd.",
            file=sys.stderr,
        )
        return 2

    # Parse field selection
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    # Exclude common VTK-generated internal arrays by default so comparisons
    # remain stable across different VTK writers/readers and when comparing
    # cell data.
    default_excludes = [
        "vtkGhostType",
        "vtkProcessId",
        "vtkOriginalPointIds",
        "vtkOriginalCellIds",
    ]
    user_excludes = [f.strip() for f in args.exclude_fields.split(",") if f.strip()]
    excludes = list(dict.fromkeys(default_excludes + user_excludes))

    overall_ok = True
    all_results: List[Dict] = []

    for i, (f0, f1) in enumerate(pairs):
        if not args.quiet and len(pairs) > 1:
            print(f"\n=== Comparison {i}: {f0}  vs  {f1} ===")

        if not os.path.exists(f0):
            print(f"ERROR: Referenced file does not exist: {f0}", file=sys.stderr)
            return 2
        if not os.path.exists(f1):
            print(f"ERROR: Referenced file does not exist: {f1}", file=sys.stderr)
            return 2

        try:
            ok, results = compare_files(
                f0,
                f1,
                fields=fields,
                exclude=excludes,
                rtol=args.rtol,
                atol=args.atol,
                compare_cells=args.compare_cells,
                quiet=args.quiet,
            )
        except Exception as e:
            print(
                f"ERROR comparing files '{f0}' and '{f1}': {e}",
                file=sys.stderr,
            )
            return 2

        overall_ok = overall_ok and ok
        all_results.append({"file0": f0, "file1": f1, "results": results})

    if args.json_out:
        with open(args.json_out, "w") as f:
            if len(all_results) == 1:
                # Backwards-compatible: emit the single comparison's results only
                json.dump(all_results[0]["results"], f, indent=2)
            else:
                json.dump(
                    {"overall_ok": overall_ok, "comparisons": all_results},
                    f,
                    indent=2,
                )

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
