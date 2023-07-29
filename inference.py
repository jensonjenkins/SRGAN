from SRGAN import compile_gen
from utils import compare_generated, save_img

generator = compile_gen()

im_name = "purple"

im_gen = compare_generated(im_path=f"sample_images/{im_name}.jpg", generator=generator)

save_img(im_path=f"generated_samples/{im_name}_sr.jpg", img=im_gen)