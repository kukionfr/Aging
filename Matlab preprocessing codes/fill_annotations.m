function fill_annotations(pth,pthtif,pthdata,WS,umpix,pthTA)
disp('interpolating and filling annotation datapoints')
% indices=[layer# annotation# x y]

outpth=[pth,'view_annotations\'];
if ~isfolder(outpth);mkdir(outpth);end

imlist = dir([pth,'*','xml']);

% k = [1         2         3             4          5         6          7         8          9         10]
%     [islets    ducts     bloodvessels  fat        acinus    submucosa  black     glass      PanIN     lumen fluid&blood]
%     [NO white  NO white  NO white      all white  NO white  NO white   NO white  all white  NO white  NO white]
% set 6==3 (connective tissue) and 8==6 so that set is 1:7

% PDAC
% k = [1         2              3          4          5]
%     [ducts     bloodvessels   islets     conn.tis   white]
%     [NO white  NO white       NO white   No white   ALL white]
   % PDAC > BV 1>6
   % conn.tiss last
   % 5 takes all white
% set 6==3 (connective tissue) and 8==6 so that set is 1:7

% PDAC new
% k = [1    2            3      4   5      6         7     8     9]
%     [PDAC bloodvessels islets fat acinus conn.tiss white ducts lymphocyte
resizeXY=umpix/0.5;
resizeXY=4;
num=length(WS{1});

tic;
infonm={};
infomat=[];
count=1;
for p=1:length(imlist) % for each image p
    datafile=[pthdata,imlist(p).name(1:end-3),'mat'];
    imnm=[imlist(p).name(1:end-3),'tif'];
    load(datafile,'xyout');
    if isempty(xyout);continue;end
    disp(imlist(p).name)
    
    xyout(:,3:4)=round(xyout(:,3:4)/resizeXY); % indices are already at desired resolution
    I=imread([pthtif,imnm]);
    szz=size(I(:,:,1));
    J=cell([1 num]);
     
    % find areas of image containing tissue
    if 0%exist([pthTA,imnm],'file')
        Ig0=imread([pthTA,imnm]);
        Ig0=~Ig0;
    else
        Ig0=rgb2gray(I)>200;
        Ig0=imopen(Ig0,strel('disk',7));
        Ig0=bwareaopen(Ig0,4);
    end
    Ig=find(Ig0);
%     I2=I(1:4:end,1:4:end,:);Ig2=Ig0(1:4:end,1:4:end);
%     figure(58),
%         subplot(1,3,1),imshow(I)
%         subplot(1,3,2),imagesc(Ig0),axis equal,axis off
%         subplot(1,3,3),imshowpair(I,Ig0)
%     ha2=get(gcf,'children');linkaxes(ha2);
    % interpolate annotation points to make closed objects
    for k=unique(xyout(:,1))' % for each annotation type k
        Jtmp=zeros(szz);
        bwtypek=Jtmp;
        cc=xyout(:,1)==k;
        xyz=xyout(cc,:);
        for pp=unique(xyz(:,2))'  % for each individual annotation
            if pp==0
                continue
            end
            cc=find(xyz(:,2)==pp);
            
            xyv=[xyz(cc,3:4); xyz(cc(1),3:4)];
            dxyv=sqrt(sum((xyv(2:end,:)-xyv(1:end-1,:)).^2,2));

            xyv(dxyv==0,:)=[]; % remove the repeating points
            dxyv(dxyv==0)=[];
            dxyv=[0;dxyv];

            ssd=cumsum(dxyv);
            ss0=1:0.49:ceil(max(ssd)); % increase by <0.5 to avoid rounding gaps
            xnew=interp1(ssd,xyv(:,1),ss0);
            ynew=interp1(ssd,xyv(:,2),ss0);
            xnew=round(xnew);
            ynew=round(ynew);
            try
                indnew=sub2ind(szz,ynew,xnew);
            catch
                disp('acds')
            end
            indnew(isnan(indnew))=[];
            bwtypek(indnew)=1;
        end
        bwtypek=imfill(bwtypek>0,'holes');
        Jtmp(bwtypek==1)=k;
        Jtmp(1:401,:)=0;Jtmp(:,1:401)=0;
        Jtmp(end-401:end,:)=0;Jtmp(:,end-401:end)=0;
        J{k}=find(Jtmp==k);
    end
    clearvars bwtypek Jtmp I xyout xyz
    % format annotations to keep or remove whitespace
    [J,ind]=format_white(J,Ig,p,WS,szz);
    
%     figure,
%     subplot(1,2,1),imshow(I)
%     subplot(1,2,2),imagesc(J),axis equal,axis off
%     ha2=get(gcf,'children');linkaxes(ha2)
    
    if contains(imnm,'TC_082')
        J(J==14)=9;
        J(J==13)=6;
    end
    
    imwrite(uint8(J),[outpth,imnm]);
    save(datafile,'ind','-append');
    
    classnum=length(unique(WS{3}));
    infonm{count,1}=imlist(p).name(1:end-4);
    try infomat(count,1:classnum)=histcounts(ind(:,2),1:length(unique(WS{3}))+1);
    catch; infomat(count,1:classnum)=zeros([1 classnum]);end
    count=count+1;
    toc;
end
save([pthdata,'info.mat'],'infonm','infomat','classnum');
disp([ [1:size(infomat,1)]' infomat ])
end

function [Jws,ind]=format_white(J0,Ig,p,WS,szz)
    ws=WS{1};       % defines keep or delete whitespace
    wsa=WS{2};      % defines non-tissue label
    wsnew=WS{3};    % redefines CNN label names
    wsorder=WS{4};  % gives order of annotations
    wsdelete=WS{5}; % lists annotations to delete
    
    Jws=zeros(szz);
    ind=[];
   % remove white pixels from annotations areas
    for k=wsorder
        if intersect(wsdelete,k)>0;continue;end % delete unwanted annotation layers
        ii=J0{k};
        iiNW=setdiff(ii,Ig);   % indices that are not white
        iiW=intersect(ii,Ig);   % indices that are white
        if ws(k)==0     % remove whitespace and add to wsa
           Jws(iiNW)=k;
           Jws(iiW)=wsa;
        elseif ws(k)==1 % keep only whitespace
           Jws(iiW)=k;
        elseif ws(k)==2 % keep both whitespace and non whitespace
           Jws(iiNW)=k;
           Jws(iiW)=k;
        end
    end

    % remove small objects and redefine labels (combine labels if desired)
    J=zeros(szz);
    for k=1:max(Jws(:))
        if wsnew(k)==wsa;dll=20;else;dll=5;end
        tmp=bwareaopen(Jws==k,dll);
        ii=find(tmp==1);
        J(tmp)=wsnew(k);
        P=[ones([length(ii) 2]).*[p wsnew(k)] ii];
        ind=cat(1,ind,P);
    end
end
