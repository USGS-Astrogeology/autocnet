from plio.io.io_gdal import GeoDataset
from pysis import sugar as isis

def pixel_to_latlon(img, sample, line):
  if isinstance(img, str):
    img = GeoDataset(img)

  driver = img.dataset.GetDriver().ShortName

  if driver == "ISIS3":
    return isis.image_to_ground(img.file_name, sample=sample, line=line)
  else:
    return img.pixel_to_latlon(x=sample, y=line)


def latlon_to_pixel(img, lat, lon):
  """
  Convert from lat/lon space to pixel space (i.e. sample/line).

  Parameters
  ----------
  lat: float
       Latitude to be transformed.

  lon : float
        Longitude to be transformed.

  Returns
  -------
     sample, line : tuple
  """
  if isinstance(img, str):
    img = GeoDataset(img)

  driver = img.dataset.GetDriver().ShortName

  if driver == "ISIS3":
    return isis.ground_to_image(img.file_name, lat=lat, lon=lon)
  else:
    sample, line =  img.latlon_to_pixel(lat=lat, lon=lon)
    return line, sample











