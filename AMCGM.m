function [y,flag] = AMCGM(X,groundtruth,k_nn_num,lambda)
classnum = max(groundtruth);
viewnum = length(X);   
n = length(groundtruth); 
flag = 0; 
[num,~] = size(X{1}); 
Fv = cell(1,viewnum);
Av_rep = zeros(num);  
X = X';

for i = 1 :viewnum
    for  j = 1:n
         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) );
    end
end

for v = 1:viewnum
    Xv = X{v};
    Av = constructW_PKN(Xv',k_nn_num);
    Av_rep = Av + Av_rep;    
    Lv = Ls(Av);
    temp = eig1(Lv,classnum+1,0);
    Fv{v} = temp(:,2:classnum+1);
    Fv{v} = Fv{v}./repmat(sqrt(sum(Fv{v}.^2,2)),1,classnum); 
end
  
Av_rep = 1/viewnum*Av_rep;
L_rep = Ls(Av_rep);
Y_rep = eig1(L_rep,classnum+1,0);
Yv = Y_rep(:,2:classnum+1);            
Yv = Yv./repmat(sqrt(sum(Yv.^2,2)),1,classnum); 

verp = lambda;  
NITER_outer = 20; 
for iter = 1:NITER_outer
    p = reweighted(Yv,Fv);
  
    distf = L2_distance_1(Yv',Yv');
    S = zeros(num);
    for i = 1:num                                                         
        idxa0 = 1 : num;
        dfi = distf(i,idxa0);
        ad = -(verp*dfi)/(2*lambda);
        S(i,idxa0) = EProjSimplex_new(ad);
    end

    SS = (S+S')/2;
    L = Ls(SS);
    Fs = zeros(size(Fv{1},1));
    for v = 1: viewnum
        Fs = Fs+2*p{v}*Fv{v}*Fv{v}';
    end
    Lm = Fs/verp;
    Yv = eig1(L-Lm, classnum, 0);
    Yv = Yv(:,1:classnum);
    [F, temp, ev] = eig1(L, classnum, 0);
    
    thre = 1e-11;
    fn1 = sum(ev(1:classnum));       
    fn2 = sum(ev(1:classnum+1));
    if fn1 > thre
       verp = 2*verp;
    elseif fn2 < thre
       verp = verp/2;  
    else
       break;
    end
end

[clusternum, y] = graphconncomp(sparse(SS)); y = y';

if clusternum == classnum
    flag = 1;
else
    sprintf('Can not find the correct cluster number: %f', lambda)
end
end

function L0 = Ls(A)
    S0 = A;
    S10 = (S0+S0')/2;
    D10 = diag(sum(S10));
    L0 = D10 - S10;
end

function pv = reweighted(F, Fv00)
viewnum = size(Fv00,2);
for v = 1:viewnum
    Fv = Fv00{v};
    pv(v) = {1/(norm(F*F'-Fv*Fv','fro')^2+1)};
end
end