function z = getMeasurement(x)
var = 1;

H = [[1 0;
     0 1;] zeros(2,size(x,1)-2);];

z = H*x + randn(2,1)*sqrt(var);
