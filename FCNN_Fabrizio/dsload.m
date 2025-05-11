function ds=dsload(fname)

file=fopen(fname,'r');

dim=fread(file,[1 2],'integer*4');

ds=fread(file,[dim(1) dim(2)],'float');

ds=ds';

