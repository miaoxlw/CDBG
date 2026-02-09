function [F, G, label1, label2, iter_num, obj0, obj] = CDBG(D,F0,G0,lam)
% Input
% D is the distance matrix from anchors to samples n*m
% F0 is the initial label matrix of n samples n*c
% G0 is the initial label matrix of m anchors m*c
% Output
% label1 is the label vector of n samples n*1
% label2 is the label vector of m anchors m*1
% F is the label matrix of n samples n*c
% G is the label matrix of m anchors m*c
% iter_num is the number of iteration
% obj is the objective function value
%
% Coded by Qimin Liang

if nargin < 5
    lam = 2;
end
 
[n,m] = size(D);  % get row of D 
[~,c] = size(F0);  % get column of F0
last1 = 0;
last2 = 0;
iter_num = 0;
obj(1) = 0;
%% compute Initial objective function value
for l = 1:c
    fl = F0(:, l);  
    gl = G0(:, l);  
    Dgl = D * gl;
    numerator = fl' * Dgl;
    denominator = (fl' * fl) * (gl' * gl);
    obj(1) = obj(1) + numerator / (denominator)^lam;
end
obj0 = obj(1);
%% store once
F = F0;
G = G0;
ftDg = zeros(1, c);
ftf = zeros(1, c);
gtg = zeros(1, c); 
ftfgtg = zeros(1, c);
ftfgtglam = zeros(1, c);
[~, label1] = max(F, [], 2);
[~, label2] = max(G, [], 2);

for k = 1:c
    ftDg(k) = F(:, k)' * D * G(:, k);
    ftf(k) = sum(F(:, k));  
    gtg(k) = sum(G(:, k)); 
    ftfgtg(k) = ftf(k) * gtg(k);
    ftfgtglam(k) = (ftf(k) * gtg(k))^lam;
end


while any(label1 ~= last1) && any(label2 ~= last2)
    last1 = label1;
    last2 = label2;

    %% 1.Fix G，Update F 
    DG = D * G;
    for i = 1:n
        p = label1(i) ;
        if ftf(p) == 1
            continue;
        end
        V1 = zeros(1, c);
        V2 = zeros(1, c);
        delta = zeros(1, c);
        for k = 1:c
            if k == p
                V1(k) = ftDg(k) - DG(i,k);
                delta(k) = ftDg(k) / ((ftfgtg(k))^lam) - V1(k) / ((ftfgtg(k) - gtg(k))^lam);
            else
                V2(k) = ftDg(k) + DG(i,k);
                delta(k) = V2(k) / ((ftfgtg(k) + gtg(k))^lam) - ftDg(k) / ((ftfgtg(k))^lam);
            end
        end
        [~,q] = min(delta);
        if p~=q
            ftDg(p) = V1(p);
            ftDg(q) = V2(q);
            ftf(p) = ftf(p) - 1;
            ftf(q) = ftf(q) + 1;
            ftfgtg(p) = ftfgtg(p) - gtg(p);
            ftfgtg(q) = ftfgtg(q) + gtg(q); 
            label1(i) = q;
            F(i, :) = 0;
            F(i, q) = 1; 
        end

    end

    %% 2.Fix F，Update G 
    FD = F' * D; 
    for i = 1:m
        p = label2(i) ;
        if gtg(p) == 1
            continue;
        end
        V1 = zeros(1, c);
        V2 = zeros(1, c);
        delta = zeros(1, c);
        for k = 1:c
            if k == p
                V1(k) = ftDg(k) - FD(k,i);
                delta(k) = ftDg(k) / ((ftfgtg(k))^lam) - V1(k) / ((ftfgtg(k) - ftf(k))^lam);
            else
                V2(k) = ftDg(k) + FD(k,i);
                delta(k) = V2(k) / ((ftfgtg(k) + ftf(k))^lam) - ftDg(k) / ((ftfgtg(k))^lam);
            end
        end
        [~,q] = min(delta);
        if p~=q
            ftDg(p) = V1(p);
            ftDg(q) = V2(q);
            gtg(p) = gtg(p) - 1;
            gtg(q) = gtg(q) + 1;
            ftfgtg(p) = ftfgtg(p) - ftf(p);
            ftfgtg(q) = ftfgtg(q) + ftf(q);
            label2(i) = q;
            G(i, :) = 0;
            G(i, q) = 1; 
        end 
    end
    iter_num = iter_num+1;
    obj(iter_num) = 0;
    %% compute objective function value
    for l = 1:c
        obj(iter_num) = obj(iter_num) + ftDg(l) / ((ftfgtg(l))^lam); %  objective function value
    end
end
end
