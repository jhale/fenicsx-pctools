import gmsh
import numpy as np


def _structured_mesh_setter(Rb, Lb, Rc, Lc, recombined, level):
    model = gmsh.model
    occ = model.occ

    model_name = "capillary-S"
    model_name += "Q" if recombined else "T"
    model.add(model_name)

    # Characteristic lengths
    h_min = 1.0e-01 * Rc * (0.5 ** level)  # min element size
    h_max = 1.0e-01 * Rb * (0.5 ** level)  # max element size
    h_avg = h_min + 0.55 * (h_max - h_min)
    ubref = 0.40  # percentage of the barrel to be refined upstream from the capillary
    if h_min / h_max < 0.1:
        print(f"Maximum element ratio may be too large: h_max / h_min = {h_max /h_min:g}")

    # Create geometry
    p1 = occ.addPoint(0.0, 0.0, 0.0)
    p2 = occ.addPoint(0.0, Rc, 0.0)
    p3 = occ.addPoint(0.0, Rb, 0.0)

    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)

    ov = occ.extrude([(1, l1), (1, l2)], Lb - ubref * Lb, 0.0, 0.0)
    l5, s1, l3, l4, l7, s2, l4, l6 = [val[1] for val in ov]  # opp., surface, bottom, top

    ov = occ.extrude([(1, l5), (1, l7)], ubref * Lb, 0.0, 0.0)
    l10, s3, l8, l9, l12, s4, l9, l11 = [val[1] for val in ov]  # opp., surface, bottom, top

    ov = occ.extrude([(1, l10)], Lc, 0.0, 0.0)
    l15, s5, l13, l14 = [val[1] for val in ov]  # opp., surface, bottom, top
    occ.synchronize()

    # Mesh refinement
    # NOTE:
    #   Transfinite curve with default "Progression" type is defined using a geometric sequence
    #     a_n = a_0 r^{n}
    #   where
    #     r   ... progression parameter (default: 1)
    #     n   ... number of elements on the curve
    #     a_0 ... smallest element size
    #     a_n ... largest element size

    def get_transfinite_coeffs(base_length, a_0, a_n):
        if a_0 == a_n:
            r = 1.0
            n = int(round(base_length / a_0))
        else:
            r = (base_length - a_0) / (base_length - a_n)
            n = int(round(np.log(a_n / a_0) / np.log(r)))
        return r, n

    # -- capillary diameter
    r, n = get_transfinite_coeffs(Rc, h_min, h_min / 0.8)
    model.mesh.setTransfiniteCurve(l15, n + 1, "Progression", coef=-r)
    model.mesh.setTransfiniteCurve(l10, n + 1, "Progression", coef=-r)
    model.mesh.setTransfiniteCurve(l5, n + 1, "Progression", coef=-r)
    model.mesh.setTransfiniteCurve(l1, n + 1, "Progression", coef=-r)

    # -- capillary length
    r, n = get_transfinite_coeffs(Lc, h_min, h_min / 0.5)
    model.mesh.setTransfiniteCurve(l13, n + 1, "Progression", coef=r)
    model.mesh.setTransfiniteCurve(l14, n + 1, "Progression", coef=r)

    # -- barrel diameter
    r, n = get_transfinite_coeffs(Rb - Rc, h_min, h_avg)
    model.mesh.setTransfiniteCurve(l12, n + 1, "Progression", coef=r)
    model.mesh.setTransfiniteCurve(l7, n + 1, "Progression", coef=r)
    model.mesh.setTransfiniteCurve(l2, n + 1, "Progression", coef=r)

    # -- barrel lenght (refined part)
    r, n = get_transfinite_coeffs(ubref * Lb, h_min, h_avg)
    model.mesh.setTransfiniteCurve(l8, n + 1, "Progression", coef=-r)
    model.mesh.setTransfiniteCurve(l9, n + 1, "Progression", coef=-r)
    model.mesh.setTransfiniteCurve(l11, n + 1, "Progression", coef=-r)

    # -- barrel length (coarse part)
    r, n = get_transfinite_coeffs(Lb - ubref * Lb, h_max, h_max)
    model.mesh.setTransfiniteCurve(l3, n + 1, "Progression", coef=-r)
    model.mesh.setTransfiniteCurve(l4, n + 1, "Progression", coef=-r)
    model.mesh.setTransfiniteCurve(l6, n + 1, "Progression", coef=-r)

    model.mesh.setTransfiniteSurface(s1, "Left")
    model.mesh.setTransfiniteSurface(s2, "Left")
    model.mesh.setTransfiniteSurface(s3, "Left")
    model.mesh.setTransfiniteSurface(s4, "Left")
    model.mesh.setTransfiniteSurface(s5, "Left")

    # Tag boundaries
    model.addPhysicalGroup(1, [l1, l2], 1)
    model.setPhysicalName(1, 1, "inlet")

    model.addPhysicalGroup(1, [l15], 2)
    model.setPhysicalName(1, 2, "outlet")

    model.addPhysicalGroup(1, [l3, l8, l13], 3)
    model.setPhysicalName(1, 3, "symmetry")

    model.addPhysicalGroup(1, [l6, l11], 4)
    model.setPhysicalName(1, 4, "bwall_hor")

    model.addPhysicalGroup(1, [l12], 5)
    model.setPhysicalName(1, 5, "bwall_ver")

    model.addPhysicalGroup(1, [l14], 6)
    model.setPhysicalName(1, 6, "cwall")

    if recombined:
        for s in [s1, s2, s3, s4, s5]:
            model.mesh.setRecombine(2, s)


def _unstructured_mesh_setter(Rb, Lb, Rc, Lc, recombined, level):
    model = gmsh.model
    occ = model.occ

    model_name = "capillary-U"
    model_name += "Q" if recombined else "T"
    model.add(model_name)

    # Characteristic lengths
    h_min = 1.0e-01 * Rc * (0.5 ** level)  # min element size
    h_max = 1.0e-01 * Rb * (0.5 ** level)  # max element size
    h_avg = h_min + 0.25 * (h_max - h_min)
    ubref = 0.40  # percentage of the barrel to be refined upstream from the capillary

    # Create geometry
    p1 = occ.addPoint(0.0, 0.0, 0.0)
    p2 = occ.addPoint(0.0, Rb, 0.0)
    p3 = occ.addPoint(Lb, Rb, 0.0)
    p4 = occ.addPoint(Lb, Rc, 0.0)
    p5 = occ.addPoint(Lb + Lc, Rc, 0.0)
    p6 = occ.addPoint(Lb + Lc, 0.0, 0.0)
    p7 = occ.addPoint(Lb, 0.0, 0.0)

    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p4)
    l4 = occ.addLine(p4, p5)
    l5 = occ.addLine(p5, p6)
    l6 = occ.addLine(p6, p7)
    l7 = occ.addLine(p7, p1)

    s1 = occ.addPlaneSurface([occ.addCurveLoop([l1, l2, l3, l4, l5, l6, l7])])
    occ.synchronize()

    # Mesh refinement
    model.mesh.setSize([(0, p1), (0, p2), (0, p3)], h_max)
    model.mesh.setSize([(0, p4), (0, p5), (0, p6), (0, p7)], h_min)
    model.mesh.field.add("Box", 1)
    model.mesh.field.setNumber(1, "XMin", Lb - ubref * Lb)
    model.mesh.field.setNumber(1, "XMax", Lb)
    model.mesh.field.setNumber(1, "YMin", 0.0)
    model.mesh.field.setNumber(1, "YMax", Rb)
    model.mesh.field.setNumber(1, "VIn", h_avg)
    model.mesh.field.setNumber(1, "VOut", h_max)
    model.mesh.field.setNumber(1, "Thickness", 0.3 * (Lb - ubref * Lb))  # outside trans. layer

    # Tag boundaries
    model.addPhysicalGroup(1, [l1], 1)
    model.setPhysicalName(1, 1, "inlet")

    model.addPhysicalGroup(1, [l5], 2)
    model.setPhysicalName(1, 2, "outlet")

    model.addPhysicalGroup(1, [l6, l7], 3)
    model.setPhysicalName(1, 3, "symmetry")

    model.addPhysicalGroup(1, [l2], 4)
    model.setPhysicalName(1, 4, "bwall_hor")

    model.addPhysicalGroup(1, [l3], 5)
    model.setPhysicalName(1, 5, "bwall_ver")

    model.addPhysicalGroup(1, [l4], 6)
    model.setPhysicalName(1, 6, "cwall")

    if recombined:
        model.mesh.setRecombine(2, s1)


def model_setter(Rb, Lb, Rc, Lc, structured, recombined, level):
    assert Rb > Rc
    assert Lb >= Lc

    if structured:
        _structured_mesh_setter(Rb, Lb, Rc, Lc, recombined, level)
    else:
        _unstructured_mesh_setter(Rb, Lb, Rc, Lc, recombined, level)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script generating 2D mesh for an axisymmetric capillary rheometer."
    )
    parser.add_argument("-Rb", type=float, default=7.5e-03, help="radius of the barrel [m]")
    parser.add_argument("-Lb", type=float, default=30.0e-03, help="length of the barrel [m]")
    parser.add_argument("-Rc", type=float, default=1.875e-03, help="radius of the capillary [m]")
    parser.add_argument("-Lc", type=float, default=15.0e-03, help="length of the capillary [m]")
    parser.add_argument("-l", type=int, dest="level", default=0, help="level of refinement")
    parser.add_argument(
        "-i", action="store_true", dest="interactive", help="run Gmsh's GUI for the geometry"
    )
    parser.add_argument(
        "-s", action="store_true", dest="structured", help="generate structured mesh"
    )
    parser.add_argument(
        "-r", action="store_true", dest="recombined", help="get mesh formed by quads"
    )
    args = parser.parse_args()

    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)

    model_setter(args.Rb, args.Lb, args.Rc, args.Lc, args.structured, args.recombined, args.level)

    if not args.structured:
        gmsh.model.mesh.field.setAsBackgroundMesh(1)

    if args.interactive:
        gmsh.fltk.run()

    gmsh.finalize()
