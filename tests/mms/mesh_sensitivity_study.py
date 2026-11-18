#!/usr/bin/env python3
"""Plot retained CTest mesh-sensitivity evidence without rerunning the operator."""
from mms_paths import REPOSITORY_ROOT
import argparse
import json
from pathlib import Path
import numpy as np

MESHES = ('uniform', 'graded', 'rotated', 'sheared', 'jitter17', 'jitter91')
NAMES = ('Uniform', 'Graded', 'Rotated 27°', 'Sheared', 'Jitter seed 17', 'Jitter seed 91')


def summarize(reports):
    records = [record for report in reports for record in report['records']]
    keys = [(r['physics'], r['mesh'], r['degree'], r['cells']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate mesh-sensitivity cases')
    groups = []
    for physics in dict.fromkeys(r['physics'] for r in records):
        for mesh in MESHES:
            for degree in (2, 3):
                group = sorted((r for r in records if (r['physics'], r['mesh'], r['degree'])
                                == (physics, mesh, degree)), key=lambda r:r['cells'])
                if len(group) != 3:
                    continue
                errors = np.array([r['solution_l2'] for r in group])
                h = np.array([r['h'] for r in group])
                orders = np.log(errors[:-1]/errors[1:])/np.log(h[:-1]/h[1:])[:, None]
                baseline = next(r for r in records if (r['physics'], r['mesh'], r['degree'], r['cells'])
                                == (physics, 'uniform', degree, group[-1]['cells']))
                entry = dict(physics=physics, mesh=mesh, degree=degree,
                             cells=[r['cells'] for r in group], solution_l2=errors.tolist(),
                             observed_orders=orders.tolist(),
                             finest_error_amplification=(errors[-1]/baseline['solution_l2']).tolist(),
                             maximum_freestream_rhs=max(r['free_stream_rhs_linf'] for r in group),
                             minimum_scaled_jacobian=min(r['minimum_scaled_jacobian'] for r in group),
                             maximum_jacobian_ratio=max(r['jacobian_ratio'] for r in group))
                if physics != 'euler':
                    gradient = np.array([r['gradient_l2'] for r in group])
                    entry['gradient_orders'] = (np.log(gradient[:-1]/gradient[1:])
                                                /np.log(h[:-1]/h[1:])[:, None]).tolist()
                groups.append(entry)
    order_flags = [issue for report in reports for issue in report['issues']
                   if issue.get('error') == 'Evolved solution order below nominal minus .5']
    control_issues = [issue for report in reports for issue in report['issues']
                      if issue.get('error') != 'Evolved solution order below nominal minus .5']
    return dict(passed=all(r['passed'] for r in reports),
                core_checks_passed=not control_issues, order_flags=order_flags,
                records=records, groups=groups,
                issues=[issue for report in reports for issue in report['issues']],
                scope='2D periodic CPG/Roe operator evolution on graded, rotated, sheared and seeded-jitter meshes; pure DG')


def generate(inputs, output):
    reports = [json.loads(path.read_text()) for path in inputs]
    if any(report.get('profile', 'extended') != 'extended' for report in reports):
        raise ValueError('Refinement plots require extended-profile reports; quick reports have one mesh level')
    summary = summarize(reports)
    output.mkdir(parents=True, exist_ok=True)
    (output/'measurements.json').write_text(json.dumps(summary, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, mesh, title in zip(axes.flat, MESHES, NAMES):
        record = next(r for r in summary['records'] if r['mesh'] == mesh and 'preview_quads' in r)
        segments = []
        for quad in record['preview_quads']:
            points = np.array(quad)
            segments.append(np.vstack((points, points[0])))
        ax.add_collection(LineCollection(segments, colors='#285a76', linewidths=.6))
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set(title=title, xlabel='x', ylabel='y')
    fig.suptitle('Actual periodic test cells: 12×12 mesh previews\nTwo reproducible jitter fields, maximum displacement 0.12 nominal cell widths per coordinate')
    fig.savefig(output/'meshes.png', dpi=180)
    fig.savefig(output/'meshes.pdf')
    plt.close(fig)
    models = tuple(dict.fromkeys(r['physics'] for r in summary['records']))
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.6*len(models)),
                            constrained_layout=True, squeeze=False)
    for ri, physics in enumerate(models):
        for ci, degree in enumerate((2, 3)):
            ax = axes[ri, ci]
            for mesh, title in zip(MESHES, NAMES):
                group = next(g for g in summary['groups'] if (g['physics'], g['mesh'], g['degree'])
                             == (physics, mesh, degree))
                h = 2./np.array(group['cells'])
                error = np.array(group['solution_l2'])[:, 0]
                ax.loglog(h, error, 'o-', label=title)
                if mesh == 'uniform':
                    ax.loglog(h, error[-1]*(h/h[-1])**(degree+1), 'k:', label=f'Expected h^{degree+1}')
            ax.set(title=f'{physics}: P{degree}', xlabel='Nominal spacing 2/N', ylabel='Density physical L2 error')
            ax.grid(True, which='both', alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('MMS evolved-solution sensitivity: real nonuniform/non-axis-aligned cells\nP2/P3 · Roe · RK4 · final time 0.05 · dt 0.00025')
    fig.savefig(output/'density-convergence.png', dpi=180)
    fig.savefig(output/'density-convergence.pdf')
    plt.close(fig)
    lines = ['# MMS mesh-sensitivity evidence', '',
             f"Operator studies: **{len(summary['records'])} cases**. Geometry/free-stream/gradient/rotation checks: **{'PASS' if summary['core_checks_passed'] else 'ISSUES FOUND'}**.", '',
             f"Strict evolved-order comparison: **{len(summary['order_flags'])} component flags**; extended fixtures return nonzero for these flags. They are retained below.", '',
             'These are 2D periodic production-operator fixtures with MFEM RK4, not a new solver launcher. '
             'Errors use element transformations and physical Gauss quadrature rather than the uniform-mesh output reconstruction.', '',
             '![Actual meshes](meshes.png)', '', '![Density convergence](density-convergence.png)', '',
             '| Physics | Mesh | P | Finest density / momentum / energy orders | Error amplification vs uniform |',
             '| --- | --- | ---: | --- | --- |']
    for g in summary['groups']:
        lines.append('| '+g['physics']+' | '+g['mesh']+' | '+str(g['degree'])+' | '
                     +' / '.join(f'{r:.3f}' for r in g['observed_orders'][-1])+' | '
                     +' / '.join(f'{r:.2f}×' for r in g['finest_error_amplification'])+' |')
    lines += ['', '## Interpretation and limits', '',
              '- Rotation is a matched orthogonal rotation of mesh, state, gradients and forcing; scalar/vector norms should agree with the uniform reference.',
              '- Grading, shear and jitter change approximation difficulty. Amplification is measured, not required to equal one.',
              '- Finest evolved-solution rates use the nominal-minus-0.5 comparison; gradient rates use p minus 0.5. Coarse rates are retained.',
              '- Cell Jacobians, physical area, free-stream RHS, positivity and error-quadrature convergence are checked.',
              '- Grading is a fixed smooth mapping; random displacements scale with h and boundaries are pinned. Seeds 17/91 are examples, not a statistical survey.',
              '- Some short-time uniform CNS P3 baseline rates also fall below the nominal comparison; compare distorted rates and error amplification against those baselines rather than assigning every flag to mesh distortion.',
              '- This covers mild 2D periodic quadrilateral distortions with one flux and the baseline mean speed. It does not establish arbitrary mesh quality, curved walls, triangles, 3D or every flux/regime.',
              '- CPU results do not establish CUDA runtime correctness; eligible builds register device fixtures.', '',
              'Full measurements and issues are in [measurements.json](measurements.json).']
    if summary['issues']:
        lines += ['', '## Observed issues', '', '```json', json.dumps(summary['issues'], indent=2), '```']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    import base64
    import html
    images = ''
    for filename in ('meshes.png', 'density-convergence.png'):
        encoded = base64.b64encode((output/filename).read_bytes()).decode()
        images += '<img style="width:100%" src="data:image/png;base64,'+encoded+'">'
    table = '<table><tr><th>Physics</th><th>Mesh</th><th>P</th><th>Finest orders: density / momentum / energy</th><th>Error amplification</th></tr>'
    for group in summary['groups']:
        table += '<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in (
            group['physics'], group['mesh'], str(group['degree']),
            ' / '.join(f'{v:.3f}' for v in group['observed_orders'][-1]),
            ' / '.join(f'{v:.2f}×' for v in group['finest_error_amplification'])))+'</tr>'
    table += '</table>'
    (output/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>MMS mesh sensitivity</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 24px;color:#18212c}'
        'td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}'
        'th{background:#eef3f8}</style><h1>MMS mesh sensitivity</h1>'
        '<p>'+str(len(summary['records']))+' extended operator cases: core checks '
        +('pass' if summary['core_checks_passed'] else 'have issues')+'. '
        'The strict evolved-order comparison retains '+str(len(summary['order_flags']))+' component flags; the extended fixtures '
        'return nonzero for those flags. Some uniform short-time CNS baselines also flag.</p>'
        +images+table+'<h2>Interpretation and limits</h2><ul>'
        +''.join('<li>'+html.escape(line[2:])+'</li>' for line in lines if line.startswith('- '))
        +'</ul><details><summary>Original extended-test issues</summary><pre>'
        +html.escape(json.dumps(summary['issues'], indent=2))+'</pre></details></html>')
    print(json.dumps(dict(passed=summary['passed'], cases=len(summary['records']), issues=summary['issues']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.input, args.output)
