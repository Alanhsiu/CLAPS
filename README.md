### Setup Git LFS
```
sudo apt-get install git-lfs
git lfs install
git lfs track "pretrained/**"

git add .gitattributes
git add pretrained/
git commit -m "Track pretrained directory with Git LFS"

git push origin main
```
