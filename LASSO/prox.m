function xnew = prox(x,H,grad,lambda) % Calculate the scaled proximal mappings

y = x - H.*grad; 

B = 1./H;
threshold = abs(y) - (lambda./B);
xnew = sign(y).*max(threshold,0);
end
