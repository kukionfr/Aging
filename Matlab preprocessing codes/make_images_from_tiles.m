% make images from tiles
pth='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\';
pthTA='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\TA\';
outpth='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\tiles\';
imlist=dir([pth,'*tif']);
sxy=256;
b=30;

pthout='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\final\classified_whitespace\';
if ~isfolder(pthout);mkdir(pthout);end
for kk=1:length(imlist)
    imnm=imlist(kk).name(1:end-4);
    pthCNN=[outpth,imnm,'\classified\'];
    if ~exist(pthCNN);continue;end
    im=imread([pth,imnm,'.tif']);
    im2=zeros(size(im(:,:,1)));
    TA=imread([pthTA,imnm,'.tif']);
    
    sz=size(im);
    count=1;
    for s1=1:256-b*3:sz(1)-sxy
        for s2=1:256-b*3:sz(2)-sxy
            try
                imtmp=imread([pthCNN,num2str(count),'.tif']);
            catch
                imtmp=zeros([256 256]);
            end
            imtmp2=imtmp(b+1:end-b,b+1:end-b,:);
            im2(s1+b:s1+b+sxy-b*2-1,s2+b:s2+b+sxy-b*2-1,:)=imtmp2;
            count=count+1;
        end
    end
    im2(~TA)=0;
    figure,
%         subplot(1,2,1),imshowpair(im,uint8(im2))
        subplot(1,2,1),imshow(im)
        subplot(1,2,2),imagesc(im2);axis equal;axis off
        ha2=get(gcf,'children');linkaxes(ha2);
    imwrite(uint8(im2),[pthout,imnm,'.tif']);
end