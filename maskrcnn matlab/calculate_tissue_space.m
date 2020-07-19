function outpth=calculate_tissue_space(pth)
% creates logical image with tissue area
disp('calculating tissue area')

imlist=dir([pth,'*tif']);

outpth=[pth,'TA\'];
if ~isfolder(outpth);mkdir(outpth);end

for kk=1:length(imlist)
    if exist([outpth,imlist(kk).name],'file');continue;end
    im=double(imread([pth,imlist(kk).name]));
    im2=im(:,:,2)<170;
    im2=bwareaopen(im2,25);
    TA=round(sum(im2(:))*100/numel(im2));
    imwrite(im2,[outpth,imlist(kk).name]);
    disp([kk length(imlist) TA])
    
    continue;
    ima=im(:,:,1);a=mode(ima(:));
    imb=im(:,:,2);b=mode(imb(:));
    imc=im(:,:,3);c=mode(imc(:));
    
    im2=im-cat(3,a,b,c);
    im2=mean(abs(im2),3)>15;
    im2=bwareaopen(im2,50);
    
    imwrite(im2,[outpth,imlist(kk).name]);
    disp([kk length(imlist)])
end