% make tiles to classify

pth='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\';
pthTA='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\TA\';
outpth='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\tiles\';

sxy=256;
b=30;

imlist=dir([pth,'*tif']);
for kk=1:length(imlist)
    imnm=imlist(kk).name(1:end-4);
    outpth2=[outpth,imnm,'\'];
    if ~isfolder(outpth2);mkdir(outpth2);end
    im=imread([pth,imnm,'.tif']);
    TA=imread([pthTA,imnm,'.tif']);
    im2=zeros(size(im));
    
    sz=size(im);
    count=1;
    for s1=1:256-b*3:sz(1)-sxy
        for s2=1:256-b*3:sz(2)-sxy
            imtmp=im(s1:s1+sxy-1,s2:s2+sxy-1,:);
            imtmpTA=TA(s1:s1+sxy-1,s2:s2+sxy-1,:);
            if sum(imtmpTA(:))<2000
                x2=imtmpTA(b+1:end-b,b+1:end-b,:);
                imtmp2=cat(3,x2,x2,x2);
            else
                imtmp2=imtmp(b+1:end-b,b+1:end-b,:);
                imwrite(imtmp,[outpth2,num2str(count),'.tif']);
            end
            im2(s1+b:s1+b+sxy-b*2-1,s2+b:s2+b+sxy-b*2-1,:)=imtmp2;
            count=count+1;
        end
    end
    disp([kk length(imlist)])
    figure(11),imshowpair(im,uint8(im2))
 end
