%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Out = ctProc(T00,dim,xymm,zmm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Convert to Density
interCept = 1024; slope = 1.00;
P = 1 + ( single(T00)*slope - interCept )/1000; P(P<0) = 0;

% Apply averaging filter
scale = 32/dim; rx = ceil(0.5*scale/xymm); rz = ceil(0.5*scale/zmm);
f = ones(2*rx+1,2*rx+1,2*rz+1); f = f/sum(f(:));
P = convn(P,f,'same');

% Resample at desired resolution
[rows,cols,levels] = size(T00);
RMM = round(rows*xymm*.1/scale); CMM = round(cols*xymm*.1/scale); ZMM = round(levels*zmm*.1/scale);
p = imresize3(P,[RMM CMM ZMM],'nearest');

if(RMM <= dim)
    bi = floor( (dim-RMM)/2 ); 
    i1 = bi+1; i2 = bi + RMM;
    pi1 = 1; pi2 = RMM;
else
    bi = floor( (RMM-dim)/2 ); 
    pi1 = bi+1; pi2 = bi + dim;
    i1 = 1; i2 = dim;
end

if(ZMM <= dim)
    bk = floor( (dim-ZMM)/2 ); 
    k1 = bk+1; k2 = bk + ZMM;
    pk1 = 1; pk2 = ZMM;
else
    bk = floor( (ZMM-dim)/2 ); 
    pk1 = bk+1; pk2 = bk + dim;
    k1 = 1; k2 = dim;
end

Out = zeros(dim,dim,dim); 
Out(i1:i2,i1:i2,k1:k2) = p(pi1:pi2,pi1:pi2,pk1:pk2);

