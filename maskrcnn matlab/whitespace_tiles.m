function whitespace_tiles(pth)

% whitespace_tiles('C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Valid\');
% whitespace_tiles('C:\Users\Noodle\Desktop\innocentive challenge\all 5x\cnn_tiles_5\normalized\Train\');

pthdata=[pth,'ANN\'];
pthhe=[pth,'HE0\'];

outpth=[pth,'HE\'];
imlist=dir([pthdata,'*tif']);
if ~isfolder(outpth);mkdir(outpth);end

% pths{1}='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\all one class\acinus\';
% pths{2}='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\all one class\ecm\';
% pths{3}='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\all one class\fat\';
% pths{4}='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\all one class\sm\';
% pths{5}='C:\Users\Noodle\Desktop\innocentive challenge\all 5x\all one class\white\';
% imls{1}=dir([pths{1},'*jpg']);
% imls{2}=dir([pths{2},'*jpg']);
% imls{3}=dir([pths{3},'*jpg']);
% imls{4}=dir([pths{4},'*jpg']);
% imls{5}=dir([pths{5},'*jpg']);
% ts=[5 6 4 3];
% nums=[1 2 3 4];
for kk=1:length(imlist)
    imnmH=strrep(imlist(kk).name,'tif','jpg');
    imh=double(imread([pthhe,imnmH]));
    im=imread([pthdata,imlist(kk).name]);
    tmp=im==0;
    tmp=cat(3,tmp,tmp,tmp);
    
    % select which background to use
    p=unique(im(:));
    
    % randomly select background type
%     bg0=randperm(4,4);bg=ts(bg0);
%     if isempty(intersect(p,bg(1)));num=nums(bg0(1));
%     elseif isempty(intersect(p,bg(2)));num=nums(bg0(2));  
%     elseif isempty(intersect(p,bg(3)));num=nums(bg0(3));  
%     elseif isempty(intersect(p,bg(4)));num=nums(bg0(4));  
%     else; num=5;
%     end
%     
%     nm=randperm(length(imls{num}),1);
%     fillz=double(imread([pths{num},imls{num}(nm).name]));
    fillz=ones([256 256 3])*241;
    imout=uint8((fillz.*tmp)+(imh.*~tmp));
    
%     figure(18),
%     subplot(1,2,1),imshow(uint8(imh))
%     subplot(1,2,2),imshow(imout)
%     disp([ts(num) p(2:end)'])

    imwrite(imout,[outpth,imnmH]);
    disp([kk length(imlist)])
end