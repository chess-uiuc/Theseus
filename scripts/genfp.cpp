#include <mfem.hpp>

using namespace mfem;

/**
 * @brief Generate the sharp flat plate 2D mesh.
 * 
 * @details This function will create an (almost) exact replica of the grid
 * used for the PC2 simulations when \p order is 1. The grid is coarsening by a
 * factor of \p order, and relevant spacings are refined by \p ref_level.
 * This is better than just doing an mfem::Mesh::UniformRefinement() because
 * growth is maintained consistently.
 * 
 * The boundary attributes are set as
 *                   
 *                            5
 *             ________________________________
 *            |                                |
 *         1  |            Domain              |  4
 *            |                                |
 *             ￣￣￣￣￣￣￣￣￣￣￣*￣￣￣￣￣￣￣￣
 *                     2                 3
 * 
 * where
 * 
 *    - 1 = supersonic inlet,
 *    - 2 = no-slip isothermal wall,
 *    - 3 = slip-wall,
 *    - 4 = zero-gradient, and
 *    - 5 = zero-gradient.
 * 
 * @note The grid order is not set to \p order. For isoparametric elements,
 * set mfem::Mesh::SetCurvature() using \p order.
 * 
 * @param order            Order of FE-space to use / factor to coarsen 
 *                         base grid.
 * @param ref_level        Refinement level to use. Must be non-negative.    
 */
Mesh MakeSharpFlatPlate2D(mfem::real_t order, int ref_level=0)
{
  // -------------------------------------------------------------------------

  /// Dimension.
  constexpr int kDim = 2;

  /// Height of the refined/unstretched region.
  constexpr double kHeight = 0.015;

  /// Length of the no-slip wall (refined/unstretched region).
  constexpr double kLength = 0.375;

  /// Total height of the domain. Large enough for weak shock to "leave".
  constexpr double kTotalHeight = 0.16;

  /// Growth rate from \ref y_wall_spacing to \ref y_spacing.
  constexpr double kYWallGrowth = 1.02;

  /// Growth rate from \ref x_le_spacing to \ref x_spacing.
  constexpr double kXLEGrowth = 1.02;

  /// Max x-spacing beyond the refined region.
  constexpr double kXStretchedSpacing = 0.01;

  /// Growth rate from \ref x_spacing to \ref kXStretchedSpacing.
  constexpr double kXStretchedGrowth = 1.08;

  /// Max y-spacing beyond the refined region.
  constexpr double kYStretchedSpacing = 0.01;

  /// Growth rate from \ref y_spacing to \ref kYStretchedSpacing.
  constexpr double kYStretchedGrowth = 1.08;

  // -------------------------------------------------------------------------

  // Compute spacings based on input ref_level + order:

  const double scale_fac = order/(ref_level+1.0);

  /// Background y-spacing in refined region.
  const double y_spacing = 1.5e-4*scale_fac;

  /// First gridpoint height from wall.
  const double y_wall_spacing = 0.75e-5*scale_fac;

  /// Background x-spacing in refined region.
  const double x_spacing = 1.5e-4*scale_fac;

  /// x-spacing at the inlet / leading edge.
  const double x_le_spacing = 1.5e-6*scale_fac;
   
  // -------------------------------------------------------------------------

  // Assemble array of x's lazily:
  std::vector<double> xs({0.0});

  double dx = x_le_spacing;
  while (dx < x_spacing)
    {
      xs.push_back(xs.back() + dx);
      dx *= kXLEGrowth;
    }
  xs.push_back(xs.back() + x_spacing);

  dx = x_spacing;
  while (xs.back() < kLength)
    {
      xs.push_back(xs.back() + dx);
    }

  // Save for setting boundary:
  const int num_plate_xs = xs.size();

  while (dx < kXStretchedSpacing*kXStretchedGrowth)
    {
      xs.push_back(xs.back() + dx);
      dx *= kXStretchedGrowth;
    }

  // -------------------------------------------------------------------------

  // Assemble array of y's lazily:
  std::vector<double> ys({0.0});

  double dy = y_wall_spacing;
  while (dy < y_spacing)
    {
      ys.push_back(ys.back() + dy);
      dy *= kYWallGrowth;
    }
  ys.push_back(ys.back() + y_spacing);

  dy = y_spacing;
  while (ys.back() < kHeight)
    {
      ys.push_back(ys.back() + dy);
    }

  while (ys.back() < kTotalHeight)
    {
      ys.push_back(ys.back() + dy);
      dy = std::min(kYStretchedSpacing, dy*kYStretchedGrowth);
    }

  // -------------------------------------------------------------------------

  // Create mesh
  const int num_vert = xs.size()*ys.size();
  const int num_elem = (xs.size()-1)*(ys.size()-1);
  const int num_bdr_elem =2*( (xs.size()-1) + (ys.size()-1) );
  Mesh mesh(kDim, num_vert, num_elem, num_bdr_elem);

  // Set vertices
  for (int j = 0; j < ys.size(); j++)
    {
      for (int i = 0; i < xs.size(); i++)
	{
	  mesh.AddVertex(xs[i], ys[j]);
	}
    }

  // Set elements
  for (int j = 0; j < ys.size() - 1; j++)
    {
      for (int i = 0; i < xs.size() - 1; i++)
	{
	  mesh.AddQuad(i   + j*xs.size(),
		       i+1 + j*xs.size(), 
		       i+1 + (j+1)*xs.size(), 
		       i   + (j+1)*xs.size());
	}
    }

  // -------------------------------------------------------------------------

  // Set boundary elements
   
  // Left + Right
  for (int j = 0; j < ys.size() - 1; j++)
    {
      // Inlet: bdr_attr 1
      mesh.AddBdrSegment(j*xs.size(), (j+1)*xs.size(), 1);

      // Outlet: bdr_attr 4
      mesh.AddBdrSegment((j+1)*xs.size() - 1, (j+2)*xs.size() - 1, 4);
    }

  // Bottom + Top
  for (int i = 0; i < xs.size() - 1; i++)
    {
      if (i < num_plate_xs - 1)
	{
	  // No-slip isothermal plate: bdr_attr 2
	  mesh.AddBdrSegment(i, i+1, 2);
	}
      else
	{
	  // Slip wall: bdr_attr 3
	  mesh.AddBdrSegment(i, i+1, 3);
	}

      // Top outlet: bdr_attr 5
      mesh.AddBdrSegment(i   + (ys.size() - 1)*xs.size(),
			 i+1 + (ys.size() - 1)*xs.size(), 
			 5);
    }

  // Finalize quad mesh, generating boundary edge elements.
  mesh.FinalizeQuadMesh(true, 0, true);

  return mesh;
}

int main(int argc, char *argv[])
{
  mfem::real_t order = 4;
  int ref_level = 0;
  bool write = true;
  bool write_compressed = true;
  bool visualization = true;
  int visport = 19916;
  char vishost[] = "localhost";

  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order", "FE order.");
  args.AddOption(&ref_level, "-r", "--ref-level", "Refinement level.");
  args.AddOption(&write, "-w", "--write", "-nw",
		 "--no-write",
		 "Enable or disable writing of mesh to file.");
  args.AddOption(&write_compressed, "-c", "--compress", "-nc",
		 "--no-compress",
		 "Enable or disable compression when writing mesh to file.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		 "--no-visualization",
		 "Enable or disable GLVis visualization.");
  args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
  args.Parse();
  if (!args.Good())
    {
      args.PrintUsage(std::cout);
      return 1;
    }
  args.PrintOptions(std::cout);

  Mesh mesh = MakeSharpFlatPlate2D(order, ref_level);

  const std::string mesh_name = "flat-plate-o" + std::to_string(order) 
    + "-r" + std::to_string(ref_level) + ".mesh";
  if (write)
    {
      
      mfem::ofgzstream ofgz(mesh_name, write_compressed);
      ofgz.precision(12);
      mesh.Print(ofgz);
    }

  if (visualization)
    {
      socketstream sock(vishost, visport);
      sock.precision(12);
      sock << "mesh\n" << mesh << std::flush;
      sock << "window_title '" << mesh_name << "'\n"
	   << "window_geometry "
	   << 10 << " " << 10 << " " << 600 << " " << 600 << "\n";
      sock << "keys " << "RA]]]" << "\n";
    }

  return 0;
}
