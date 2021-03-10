import gmsh
import numpy as np


def model_setter(H, L, R, level):
    model = gmsh.model
    occ = model.occ

    model_name = "cylinder-symmetric"
    model.add(model_name)

    # Geometry
    rectangle = occ.addRectangle(-L, 0.0, 0.0, 2.0 * L, H)
    circle = occ.addPlaneSurface([occ.addCurveLoop([occ.addCircle(0.0, 0.0, 0.0, R)])])
    channel = occ.cut([(2, rectangle)], [(2, circle)])
    channel = channel[0][0][1]  # exctract resulting surface tag
    l1, l2, l3, l4, l5, l6 = [val[1] for val in occ.getEntities(dim=1)]
    occ.synchronize()

    # Mesh refinement
    h_max = H / 6.0
    h_min = 0.0127

    # FIXME: Figure out how to do global refinements based on provided level
    if level > 0:
        raise NotImplementedError("Global refinement levels not supported at the moment")

    field_h_max = model.mesh.field.add("MathEval")
    model.mesh.field.setString(field_h_max, "F", str(h_max))
    model.mesh.field.setAsBackgroundMesh(field_h_max)

    field_dist = model.mesh.field.add("Distance")
    model.mesh.field.setNumber(field_dist, "NNodesByEdge", np.pi * R / h_min + 1)
    model.mesh.field.setNumbers(field_dist, "EdgesList", [l1])

    field_curved = model.mesh.field.add("Threshold")
    model.mesh.field.setNumber(field_curved, "IField", field_dist)
    model.mesh.field.setNumber(field_curved, "LcMin", h_min)
    model.mesh.field.setNumber(field_curved, "LcMax", h_max)
    model.mesh.field.setNumber(field_curved, "DistMin", 0.1 * (H - R))
    model.mesh.field.setNumber(field_curved, "DistMax", L / 3.0)

    field_min = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(field_min, "FieldsList", [field_h_max, field_curved])
    model.mesh.field.setAsBackgroundMesh(field_min)

    gmsh.option.setNumber("Mesh.Smoothing", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    # Physical groups
    model.addPhysicalGroup(1, [l3], 1)
    model.setPhysicalName(1, 1, "inlet")

    model.addPhysicalGroup(1, [l5], 2)
    model.setPhysicalName(1, 2, "outlet")

    model.addPhysicalGroup(1, [l4], 3)
    model.setPhysicalName(1, 3, "wall")

    model.addPhysicalGroup(1, [l2, l6], 4)
    model.setPhysicalName(1, 4, "symmetry")

    model.addPhysicalGroup(1, [l1], 5)
    model.setPhysicalName(1, 5, "cylinder")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script generating 2D mesh for symmetric flow past a cylinder."
    )
    parser.add_argument("-H", type=float, default=2.0, help="halfwidth of the channel")
    parser.add_argument("-L", type=float, default=15.0, help="length of the channel")
    parser.add_argument("-R", type=float, default=1.0, help="radius of the cylinder")
    parser.add_argument("-l", type=int, dest="level", default=0, help="level of refinement")
    parser.add_argument(
        "-i", action="store_true", dest="interactive", help="run Gmsh's GUI for the geometry"
    )
    args = parser.parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    model_setter(args.H, args.L, args.R, args.level)

    if args.interactive:
        gmsh.fltk.run()
    gmsh.finalize()
