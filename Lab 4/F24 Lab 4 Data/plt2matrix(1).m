function M = plt2matrix(filename)
%plt2matrix.m converts a tecplot output file from the OMS lite TSP/PSP
%software to a matrix
%
%Usage: M = plt2matrix(filename)
% filename is the full file path of the tecplot file
% M is the converted output matrix

x_tot=0;
y_tot=0;

%read tecplot file and convert to matrix

I=importdata(filename,' ',4);
I.data;
X=I.data(:,2);
Y=I.data(:,1);
Int=I.data(:,3);
[temp,x_tot]=max(X);
y_tot=size(X,1)/x_tot;
M(:,:)=transpose(reshape(Int,x_tot,y_tot));
M=rot90(M);

end
