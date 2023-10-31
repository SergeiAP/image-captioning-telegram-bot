# from google_images_search import GoogleImagesSearch
# from io import BytesIO
# from PIL import Image
# from data import config

# in this case we're using PIL to keep the BytesIO as an image object
# that way we don't have to wait for disk save / write times
# the image is simply kept in memory
# this example should display 3 pictures of puppies!

gis = GoogleImagesSearch(config.GCS_DEVELOPER_KEY, config.GCS_CX)

my_bytes_io = BytesIO()

gis.search({'q': 'puppies', 'num': 3})
for image in gis.results():
    # here we tell the BytesIO object to go back to address 0
    my_bytes_io.seek(0)

    # take raw image data
    raw_image_data = image.get_raw_data()

    # this function writes the raw image data to the object
    image.copy_to(my_bytes_io, raw_image_data)

    # or without the raw data which will be automatically taken
    # inside the copy_to() method
    image.copy_to(my_bytes_io)

    # we go back to address 0 again so PIL can read it from start to finish
    my_bytes_io.seek(0)

    # create a temporary image object
    temp_img = Image.open(my_bytes_io)

    # show it in the default system photo viewer
    temp_img.show()