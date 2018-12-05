function y = random_by_row_matrix(x)
% x matriz de m x n (filas x columnas)
%
rows = size(x,1); %rows = m filas
shuffle1 = randsample(rows,rows); %rows valores, sin reemplazo
                                  %valores desde 1 a rows
shuffle2 = randsample(rows,rows);

%Devuelve y = x, pero con sus filas de forma aleatoria
%uniformemente distribuidas
m = x(shuffle1,:);
y = m(shuffle2,:);

end