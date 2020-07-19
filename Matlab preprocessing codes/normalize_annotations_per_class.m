pth='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\ann\';
pthH='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\HE\';

% get number of pixels for each class
imlist=dir([pth,'*tif']);
num=zeros([7 length(imlist)]);
for kk=1:length(imlist)
   im=imread([pth,imlist(kk).name]);
   for p=1:7
       tmp=im==p;
       num(p,kk)=sum(tmp(:));
   end
end
disp('total number per class')
disp(sum(num,2)')

% randomly choose tiles to keep training
outpthtrain='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Train\ANN\';
outpthtrainF='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Train\ANN_stack\';
outpthtrainH='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Train\HE0\';
mkdir(outpthtrain);mkdir(outpthtrainH);mkdir(outpthtrainF);
c=min(sum(num,2));
tt=randperm(length(imlist),length(imlist));
kpt=ones([1 7])*length(imlist);
for kk=1:length(imlist)
    kpst=sum(num(:,tt(1:kk)),2);
    for k=find(kpst>c*0.85)'
       if kpt(k)==length(imlist);kpt(k)=kk;end
    end
end
disp('number for training')
disp(kpt)

% keep more files for validation
outpthval='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Valid\ANN\';
outpthvalF='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Valid\ANN_stack\';
outpthvalH='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Valid\HE0\';
mkdir(outpthval);mkdir(outpthvalH);mkdir(outpthvalF);
% randomly choose tiles to keep validation
kpv=ones([1 7])*length(imlist);
for kk=1:length(imlist)
    kpsv=sum(num(:,tt(1:kk)),2);
    for k=find(kpsv>c)'
       if kpv(k)==length(imlist);kpv(k)=kk;end
    end
end
disp('number for validation')
disp(kpv-kpt)

% save training images
for count=1:length(imlist)
   kk=tt(count);
   imnm=imlist(kk).name;
   imnmH=strrep(imnm,'tif','jpg');
   im0=imread([pth,imnm]);
   im=zeros(size(im0));
   for z=1:7
      if count<=kpt(z)
          im(im0==z)=z;
      end
   end
    if sum(im(:))>0
        imwrite(uint8(im),[outpthtrain,imlist(kk).name]);
        copyfile([pthH,imnmH],[outpthtrainH,imnmH]);
    end
    disp([count length(imlist)])
end

% save validation images
for count=1:length(imlist)
   kk=tt(count);
   imnm=imlist(kk).name;
   imnmH=strrep(imnm,'tif','jpg');
   im0=imread([pth,imnm]);
   im=zeros(size(im0));
   for z=1:7
      if count>kpt(z) && count<=kpv(z)
          im(im0==z)=z;
      end
   end
    if sum(im(:))>0
        imwrite(uint8(im),[outpthval,imlist(kk).name]);
        copyfile([pthH,imnmH],[outpthvalH,imnmH]);
    end
    disp([count length(imlist)])
end

% check testing
imlist=dir([outpthtrain,'*tif']);
numt=zeros([7 length(imlist)]);
for kk=1:length(imlist)
   im=imread([outpthtrain,imlist(kk).name]);
   for p=1:7
       tmp=im==p;
       numt(p,kk)=sum(tmp(:));
   end
end
disp('total number in testing')
disp(sum(numt,2)')

% check validation
imlist=dir([outpthval,'*tif']);
numv=zeros([7 length(imlist)]);
for kk=1:length(imlist)
   im=imread([outpthval,imlist(kk).name]);
   for p=1:7
       tmp=im==p;
       numv(p,kk)=sum(tmp(:));
   end
end
disp('total number in validation')
disp(sum(numv,2)')
