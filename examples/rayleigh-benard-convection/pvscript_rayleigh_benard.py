import os
import sys

import numpy as np

TEXMFDIST_DIR = os.getenv("TEXMFDIST_DIR")
_special_font = f"{TEXMFDIST_DIR}/fonts/truetype/google/tinos/Tinos-Regular.ttf"

# check the number of input arguments
assert len(sys.argv[1:]) == 4
output_file = sys.argv[1]

# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Xdmf3ReaderT'
field_vxdmf = Xdmf3ReaderT(FileName=[sys.argv[2]])
field_vxdmf.PointArrays = ["v"]

# create a new 'Xdmf3ReaderT'
field_pxdmf = Xdmf3ReaderT(FileName=[sys.argv[3]])
field_pxdmf.PointArrays = ["p"]

# create a new 'Xdmf3ReaderT'
field_Txdmf = Xdmf3ReaderT(FileName=[sys.argv[4]])
field_Txdmf.PointArrays = ["T"]

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")
# uncomment following to set a specific view size
renderView1.ViewSize = [1216, 736]

# show/hide orientation axes
renderView1.OrientationAxesVisibility = 1
# renderView1.OrientationAxesInteractivity = 1
# renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]

# # add axes
# renderView1.AxesGrid.Visibility = 1
# renderView1.AxesGrid.XTitle = 'x'
# renderView1.AxesGrid.YTitle = 'y'
# renderView1.AxesGrid.ZTitle = 'z'
# renderView1.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
# renderView1.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
# renderView1.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
# renderView1.AxesGrid.GridColor = [0.0, 0.0, 0.0]
# renderView1.AxesGrid.XAxisUseCustomLabels = 1
# renderView1.AxesGrid.YAxisUseCustomLabels = 1
# renderView1.AxesGrid.ZAxisUseCustomLabels = 1

# --------------------------------------------------------------------------------------------------
# VELOCITY FIELD

# show data in view
field_vxdmfDisplay = Show(field_vxdmf, renderView1, "UnstructuredGridRepresentation")

# get color transfer function/color map for v
vLUT = GetColorTransferFunction("v")
vLUT.ApplyPreset("Viridis (matplotlib)", True)

# set scalar coloring
ColorBy(field_vxdmfDisplay, ("POINTS", "v", "Magnitude"))

# rescale color and/or opacity maps used to include current data range
field_vxdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
field_vxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color legend/bar for vLUT in view renderView1
vLUTColorBar = GetScalarBar(vLUT, renderView1)

# Properties modified on vLUTColorBar
vLUTColorBar.Title = "velocity"
vLUTColorBar.ComponentTitle = "magnitude"
vLUTColorBar.TitleFontFamily = "File"
vLUTColorBar.TitleFontFile = _special_font
vLUTColorBar.LabelFontFamily = "File"
vLUTColorBar.LabelFontFile = _special_font

# create a new 'Stream Tracer'
streamTracer1 = StreamTracer(Input=field_vxdmf, SeedType="Point Source")
streamTracer1.Vectors = ["POINTS", "v"]
streamTracer1.SeedType.NumberOfPoints = 75
streamTracer1.SeedType.Radius = 0.5

# show data in view
streamTracer1Display = Show(streamTracer1, renderView1, "GeometryRepresentation")

# create a new 'Tube'
tube1 = Tube(Input=streamTracer1)
tube1.Scalars = ["POINTS", "AngularVelocity"]
tube1.Vectors = ["POINTS", "Normals"]
tube1.Radius = 0.008

# show data in view
tube1Display = Show(tube1, renderView1, "GeometryRepresentation")

# hide data in view
Hide(field_vxdmf, renderView1)
Hide(streamTracer1, renderView1)

# --------------------------------------------------------------------------------------------------
# PRESSURE

# show data in view
field_pxdmfDisplay = Show(field_pxdmf, renderView1, "UnstructuredGridRepresentation")
field_pxdmfDisplay.Opacity = 0.7

# get color transfer function/color map for p
pLUT = GetColorTransferFunction("p")
pLUT.ApplyPreset("Cool to Warm (Extended)", True)

# show color bar/color legend
field_pxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color legend/bar for pLUT in view renderView1
pLUTColorBar = GetScalarBar(pLUT, renderView1)

# Properties modified on pLUTColorBar
pLUTColorBar.TitleFontFamily = "File"
pLUTColorBar.TitleFontFile = _special_font
pLUTColorBar.LabelFontFamily = "File"
pLUTColorBar.LabelFontFile = _special_font

# Properties modified on pLUTColorBar
pLUTColorBar.Title = "pressure"

# create pressure contours
p_range = field_pxdmf.PointData.GetArray("p").GetRange()
contour1 = Contour(Input=field_pxdmf)
contour1.ContourBy = ["POINTS", "p"]
contour1.Isosurfaces = np.linspace(*p_range, 17)
contour1.PointMergeMethod = "Uniform Binning"
contour1Display = Show(contour1, renderView1, "GeometryRepresentation")
contour1Display.Opacity = 0.6
Hide(field_pxdmf, renderView1)

# --------------------------------------------------------------------------------------------------
# TEMPERATURE

# show data in view
field_TxdmfDisplay = Show(field_Txdmf, renderView1, "UnstructuredGridRepresentation")

# change representation type
field_TxdmfDisplay.SetRepresentationType("Feature Edges")

# get color transfer function/color map for p
tLUT = GetColorTransferFunction("T")
tLUT.ApplyPreset("Cool to Warm", True)

# show color bar/color legend
field_TxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color legend/bar for tLUT in view renderView1
tLUTColorBar = GetScalarBar(tLUT, renderView1)

# Properties modified on pLUTColorBar
tLUTColorBar.TitleFontFamily = "File"
tLUTColorBar.TitleFontFile = _special_font
tLUTColorBar.LabelFontFamily = "File"
tLUTColorBar.LabelFontFile = _special_font

# Properties modified on pLUTColorBar
tLUTColorBar.Title = "temperature"

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(Input=field_Txdmf)

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1, "GeometryRepresentation")

# create a new 'Clip'
clip1 = Clip(Input=extractSurface1)
clip1.ClipType = "Plane"
clip1.HyperTreeGridClipper = "Plane"
clip1.Scalars = [None, ""]

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.5, 1.0 - 1e-4, 0.5]
clip1.ClipType.Normal = [0.0, -1.0, 0.0]

# Properties modified on clip1
clip1.Scalars = ["POINTS", ""]

# show data in view
clip1Display = Show(clip1, renderView1, "UnstructuredGridRepresentation")

# change representation type
clip1Display.SetRepresentationType("Wireframe")

# hide data in view
Hide(extractSurface1, renderView1)

# --------------------------------------------------------------------------------------------------

# current camera placement for renderView1
renderView1.CameraPosition = [2.1143223764414962, -2.0659124886854543, 1.9206458692768749]
renderView1.CameraFocalPoint = [0.5303014110198067, 0.4603188663305029, 0.4023622512577075]
renderView1.CameraViewUp = [-0.25766476173215325, 0.3739075639854851, 0.8909556690183599]
renderView1.CameraParallelScale = 0.8660254037844386

# save screenshot
SaveScreenshot(
    output_file, renderView1, ImageResolution=[1125, 736], OverrideColorPalette="WhiteBackground",
)

#### uncomment the following to render all views
# RenderAllViews()
