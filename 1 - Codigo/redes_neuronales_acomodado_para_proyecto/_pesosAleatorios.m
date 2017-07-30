function W = _pesosAleatorios(l_out, l_in)
  INIT_EPSILON = sqrt(6)/sqrt(l_out+l_in);

  W = rand(l_out, l_in)*(2*INIT_EPSILON) - INIT_EPSILON;
endfunction
