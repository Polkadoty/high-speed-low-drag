function [P0, Pi, P1, P2, P3, P4] = windTunnelP(volts, lower, upper)

P0V = volts(:, 2);
PiV = volts(:, 3);
P1V = volts(:, 4);
P2V = volts(:, 6);
P3V = volts(:, 7);
P4V = volts(:, 5);

avgP0V = mean(P0V(lower:upper));
avgPiV = mean(PiV(lower:upper));
avgP1V = mean(P1V(lower:upper));
avgP2V = mean(P2V(lower:upper));
avgP3V = mean(P3V(lower:upper));
avgP4V = mean(P4V(lower:upper));

P0psi = (60/0.1)*avgP0V;
Pipsi = (15/0.1)*avgPiV;
P1psi = (15/0.1)*avgP1V;
P2psi = (15/0.1)*avgP2V;
P3psi = (15/0.1)*avgP3V;
P4psi = (15/0.1)*avgP4V;

P0 = P0psi*6894.76 + 101325;
Pi = Pipsi*6894.76 + 101325;
P1 = P1psi*6894.76 + 101325;
P2 = P2psi*6894.76 + 101325;
P3 = P3psi*6894.76 + 101325;
P4 = P4psi*6894.76 + 101325;