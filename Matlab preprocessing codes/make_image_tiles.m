function make_image_tiles(pth,pthtiles)
if nargin==0
    pth='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\';
    pthtiles='C:\Users\Noodle\Desktop\innocentive challenge\view_annotations\';
end

imlist=dir([pth,'*tif']);
outpth=[pth,'cnn_tiles_5\HE\'];if ~isfolder(outpth);mkdir(outpth);end
outpthTA=[pth,'cnn_tiles_5\ann\'];if ~isfolder(outpthTA);mkdir(outpthTA);end

sxy=256;
count=1;
for kk=1:length(imlist)
    imnm=imlist(kk).name;
    im=imread([pth,imnm]); 
    TA0=imread([pthtiles,imnm]);
    TA=TA0;

    
    sz=size(TA);
    for s1=500:256:sz(1)-500
        for s2=500:256:sz(2)-500
            tmp=TA(s1:s1+sxy-1,s2:s2+sxy-1,:)>0;
            if sum(tmp(:))/numel(tmp)<0.05;continue;end % must be 4% annotation
            imout=im(s1:s1+sxy-1,s2:s2+sxy-1,:);
            TAout=TA(s1:s1+sxy-1,s2:s2+sxy-1,:);
            imwrite(imout,[outpth,num2str(count),'.jpg']);
            imwrite(uint8(TAout),[outpthTA,num2str(count),'.tif']);
            count=count+1;
        end
    end
    disp([kk length(imlist) count])
end

