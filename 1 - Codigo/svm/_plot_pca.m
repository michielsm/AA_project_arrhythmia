function _plot_pca(z, Y)

number_instances = rows(z);

for i=1:number_instances
  if(Y(i) == 1)
    scatter(z(i, 1), z(i, 2), 10, "blue", "filled");
  endif;
  
  if(Y(i) == 2)
    scatter(z(i, 1), z(i, 2), 10,  "red", "filled");
  endif;
  
  if(Y(i) == 3)
    scatter(z(i, 1), z(i, 2), 10,  "black", "filled");
  endif;
  
  if(Y(i) == 4)
    scatter(z(i, 1), z(i, 2), 10,  "yellow", "filled");
  endif;

  if(Y(i) == 5)
    scatter(z(i, 1), z(i, 2), 10,  "magenta", "filled");
  endif;
    
  if(Y(i) == 6)
    scatter(z(i, 1), z(i, 2), 10,  [126 26 125] ./ 255, "filled");
  endif;
    
  if(Y(i) == 7)
    scatter(z(i, 1), z(i, 2), 10,  [53 178 82] ./ 255, "filled");
  endif;
    
  if(Y(i) == 8)
    scatter(z(i, 1), z(i, 2), 10,  [100 038 32] ./ 255, "filled");
  endif;
    
  if(Y(i) == 9)
    scatter(z(i, 1), z(i, 2), 10,  [233 31 23] ./ 255, "filled");
  endif;
    
  if(Y(i) == 10)
    scatter(z(i, 1), z(i, 2), 10,  [121 98 121] ./ 255, "filled");
  endif;
    
  if(Y(i) == 11)
    scatter(z(i, 1), z(i, 2), 10,  "black", "filled");
  endif;
    
  if(Y(i) == 12)
    scatter(z(i, 1), z(i, 2), 10,  [012 100 211] ./ 255, "filled");
  endif;
  
  if(Y(i) == 13)
    scatter(z(i, 1), z(i, 2), 10,  [31 18 12] ./ 255, "filled");
  endif;

  hold on;
endfor;
xlabel("Principal component 1", "fontsize", 11);
ylabel("Principal component 2", "fontsize", 11);

endfunction