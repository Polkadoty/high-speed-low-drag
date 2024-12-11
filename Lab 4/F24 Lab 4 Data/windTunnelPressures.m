function [P0, Pi] = windTunnelPressures(data, lower, upper)

P0V = data(:, 3);
PiV = data(:, 4);

avgP0V = mean(P0V(lower:upper));
avgPV = mean(PiV(lower:upper));

P0psi = 4137*avgP0V;
Pipsi = 1034*avgPV;

P0 = P0psi + 101.325;
Pi = Pipsi + 101.325;
