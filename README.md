# SRGAN

A generative adversarial network (GAN) based on the SRGAN architecture from the 2016 paper, "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" by C. Ledig et al. 

## About
- Takes in an image of 32x32 pixels and outputs an image with an increased resolution of 128x128 pixels.
- Trained using the <a href="https://press.liacs.nl/mirflickr/mirdownload.html" target="_blank">MIRFLICKR</a> dataset.
  
## How to run
- Install the required dependencies
  ```bash
  pip install -r requirements.txt
  ```
- In `inference.py`, change the `im_name` variable into the designated image file name.
- Run `inference.py`.
  ```bash
  python inference.py
  ```
- The matplotlib window should pop up showing the comparison, the newly generated image is then saved in the folder `generated_samples` if the target save path is unchanged.

## Demo
<table>
  <tr>
    <td align="center">
      <img src="sample_images/rose.jpg" width="128" height="128"/>
      <br />
      Original Image (32x32)
    </td>
    <td align="center">
      <img src="generated_samples/rose_sr.jpg"  width="128" height="128"/>
      <br />
      Generated Image (128x128)
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="sample_images/purple.jpg"  width="128" height="128"/>
      <br />
      Original Image (32x32)
    </td>
    <td align="center">
      <img src="generated_samples/purple_sr.jpg"  width="128" height="128"/>
      <br />
      Generated Image (128x128)
    </td>
  </tr>
</table>




## Architecture
<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png"/>



