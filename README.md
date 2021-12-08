## about 

- this repo implements the mono version of [OrcVIO](https://moshan.cf/orcvio_githubpage/), and is developed for analyzing its covariance. 
- the original msckf is in the `msckf` branch 

## environment setup 

- platform `ubuntu 18.04` 
- using conda `conda env create --file environment.yml`
- install pip packages 
   * `pip install sophuspy` 

## EuRoC demo 

- setup python path 
```
export PYTHONPATH="${PYTHONPATH}:/home/erl/orcvio/orcvio-covariance-python"
```
- perform tests 
```
python ./tests/test_msckf.py
python ./tests/test_quaternions.py
python ./tests/test_triangulation.py
python ./tests/test_twopoint_ransac.py
```
- run demo on euroc MH02 easy 
```
python ./examples/run_on_euroc.py --euroc_folder /mnt/disk2/euroc/MH_02_easy/mav0 --use_viewer --start_timestamp 1403636896901666560
```
- result
> Green is groundtruth trajectory, red is the estimated trajectory from OrcVIO-Mono 
![demo](assets/demo.gif)
- covariance analysis 
> From figure below we can see MSCKF is making overconfident estimations whereas OrcVIO does not, due to the closed-form covariance propagation. The results are similar to similar to Fig 6, 8 in [Camera-IMU-based localization: Observability analysis and consistency improvement](http://heschian.io/_files/Joel_Hesch_IJRR14.pdf) 
![demo](assets/cov-euroc.png)

## references 

- https://github.com/Edwinem/msckf_tutorial
- https://github.com/uoip/stereo_msckf

