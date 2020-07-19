function Openslide_svstotif(pth,outpths,pixelresolutions,nm,datafile)
% numcopies = how many rescaled images are required
% pth = path to images
% pixel resolutions = resizing factors for images in um/pixel
%   - should be in ascending order, i.e. pixelresolutions = [2 4 8]
%     so that images are produced in descending order im1>im2>im3,...
% will be saved in folder in pth named for magnification of image 
% assuming orginial magnification was 20x

if isempty(datafile)
    savedd=0;
else
    savedd=1;
end

if isempty(nm)
    imlist = dir([pth,'*','svs']);
    imlist2 = dir([pth,'*','ndpi']);
    imlist=[imlist;imlist2];
else
    imlist = dir([pth,'*',nm,'*']);
end

reduce_annotations=zeros([1 length(imlist)]);
% 
% path(path,'\\babyserverdw3\Ashley Kiemen\PanIN Modelling Package\openslide\');
% path(path,'\\babyserverdw3\Ashley Kiemen\PanIN Modelling Package\openslide\bin\');
openslide_load_library;


% set image output folder
if length(pixelresolutions)>1
    outpth=[pth,char(outpths(1))];
else
    outpth=[pth,outpths];
end
if ~isfolder(outpth); mkdir(outpth); end


count=0;
for k=1:length(imlist)
    if contains(imlist(k).name,'.xml')
        continue
    end
    count=count+1;
    tic
    disp(imlist(k).name)
    imNtif=strrep(imlist(k).name,'.ndpi','.tif'); 
    imNtif=strrep(imNtif,'.svs','.tif'); 
    try openslidePointer = openslide_open([pth,imlist(k).name]);
    catch
       disp('failed to open image')
       continue
    end
    [mppX, mppY, width, height, numberOfLevels,downsampleFactors,...
        objectivePower] = openslide_get_slide_properties(openslidePointer);
    reduce_annotations(count)=mppX;
    fx=pixelresolutions(1)/mppX; % resizing factor to produce image of 2um/pixel

    tilesz=ceil(2500*fx); % open image size that produces resized image of 2500x2500.
    reduce_annotations(count)=fx;

    % generate grid;; determine grid number
    width=single(width);
    height=single(height);
    Ngx=ceil(width/tilesz);
    Ngy=ceil(height/tilesz);
    readlevel=0; % readlevel=0 for reading from full resolution image tile

    imtif=zeros(ceil(height/fx),ceil(width/fx),3); % preallocate output image
    imtif=uint8(imtif);
    for kx=1:Ngx
        for ky=1:Ngy
            xid=(kx-1)*tilesz+1:kx*tilesz;
            yid=(ky-1)*tilesz+1:ky*tilesz;
            xid=xid(xid<=width);
            yid=yid(yid<=height);
            xPos=xid(1);
            rx0=length(xid)-1;
            yPos=yid(1);
            ry0=length(yid)-1;

            try % read image tile at full resolution
                [imtmp] = openslide_read_region(openslidePointer,xPos,yPos,rx0,ry0,'level',readlevel);
            catch % reload openslide in case of error
                openslidePointer = openslide_open([pth,imlist(k).name]);
                [imtmp] = openslide_read_region(openslidePointer,xPos,yPos,rx0,ry0,'level',readlevel);
                disp('CATCH: reload openslidePointer')
            end
            xx=ceil(size(imtmp(:,:,1))/fx);
            imtmp=imresize(imtmp,xx,'nearest');
            xid4x=(kx-1)*round(tilesz/fx)+1:(kx-1)*round(tilesz/fx)+size(imtmp,2);
            yid4x=(ky-1)*round(tilesz/fx)+1:(ky-1)*round(tilesz/fx)+size(imtmp,1);
            imtif(yid4x,xid4x,:)=imtmp;   
            disp([kx ky])
        end
    end          
    imtmp=squeeze(sum(imtif,3));
    x00=sum(imtmp,1)==0;
    y00=sum(imtmp,2)==0;

    imtif(:,x00,:)=[];
    imtif(y00,:,:)=[];
    im1=imtif(:,:,1);m1=mode(im1(:));
    im2=imtif(:,:,2);m2=mode(im2(:));
    im3=imtif(:,:,3);m3=mode(im3(:));
    im1(im1==0)=m1;
    im2(im2==0)=m2;
    im3(im3==0)=m3;
    imtif=cat(3,im1,im2,im3);
    disp([k length(imlist)])
    
    imwrite(imtif,[outpth,imNtif]);
    for jj=2:length(pixelresolutions)
%         outpth2=strcat(pth,char(outpths(jj)));
        outpth2=[pth,char(outpths(jj))];
        if ~isfolder(outpth2); mkdir(outpth2); end
        
        xx=pixelresolutions(1)/pixelresolutions(jj); % calculate rescale factor: ex. 2um/px --> 8um/px = rescale image to 1/4
        imtif2=imresize(imtif,xx,'nearest');
        imwrite(imtif2,[outpth2,imNtif]);
    end
    toc;
end
reduce_annotations(count+1:end)=[];

if savedd
    if exist(datafile,'file')
        save(datafile,'reduce_annotations','-append')
    else
        save(datafile,'reduce_annotations');
    end
end