dataM2A0 = readmatrix('M2AoA0_0.txt');
dataM2A3 = readmatrix('M2AoA3_0.txt');
dataM2A7 = readmatrix('M2AoA7_0.txt');

dataM225A0 = readmatrix('M225AoA0_0.txt');
dataM225A3 = readmatrix('M225AoA3_0.txt');
dataM225A7 = readmatrix('M225AoA7_0.txt');

dataM25A0 = readmatrix('M25AoA0_0.txt');
dataM25A3 = readmatrix('M25AoA3_0.txt');
dataM25A7 = readmatrix('M25AoA7_0.txt');

dataM3A0 = readmatrix('M3AoA0_0.txt');
dataM3A3 = readmatrix('M3AoA3_0.txt');
dataM3A7 = readmatrix('M3AoA7_0.txt');

%% Mach 2 AOA 0

%P0
figure(1)
plot(1:size(dataM2A0, 1), dataM2A0(:, 2))

%Pi
figure(2)
plot(1:size(dataM2A0, 1), dataM2A0(:, 3))

%% Measured Pressures

lower = 2115;
upper = 2761;

[M2A0P0, M2A0Pi, M2A0P1, M2A0P2, M2A0P3, M2A0P4] = windTunnelP(dataM2A0, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))

M2A0M = MfromP(M2A0P0, M2A0Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 1.71714191;
ratiobot = 1.71714191;

tM2A0P1 = ratiotop*M2A0Pi
tM2A0P3 = ratiobot*M2A0Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top = 1.67689681
% incoming mach bot = 1.67689681
% turn angle = 20


tM2A0P2 = 21174.37;
tM2A0P4 = 21174.37;




%% Mach 2 AOA 3

%P0
figure(1)
plot(1:size(dataM2A3, 1), dataM2A3(:, 2))

%Pi
figure(2)
plot(1:size(dataM2A3, 1), dataM2A3(:, 3))

%% Measured Pressures

lower = 2826;
upper = 3220;

[M2A3P0, M2A3Pi, M2A3P1, M2A3P2, M2A3P3, M2A3P4] = windTunnelP(dataM2A3, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M2A3M = MfromP(M2A3P0, M2A3Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 2.00259787;
ratiobot = 1.46988810;

tM2A3P1 = ratiotop*M2A3Pi
tM2A3P3 = ratiobot*M2A3Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  1.56861459
% incoming mach bot =  1.79308313
% turn angle = 20


tM2A3P2 = 24423.5;
tM2A3P4 = 16365.13;



%% Mach 2 AOA 7

%P0
figure(1)
plot(1:size(dataM2A7, 1), dataM2A7(:, 2))

%Pi
figure(2)
plot(1:size(dataM2A7, 1), dataM2A7(:, 3))

%% Measured Pressures

lower = 4467;
upper = 4887;

[M2A7P0, M2A7Pi, M2A7P1, M2A7P2, M2A7P3, M2A7P4] = windTunnelP(dataM2A7, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M2A7M = MfromP(M2A7P0, M2A7Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 2.44590202;
ratiobot = 1.18362860;

tM2A7P1 = ratiotop*M2A7Pi
tM2A7P3 = ratiobot*M2A7Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top = 1.39855387
% incoming mach bot = 1.93251348
% turn angle = 20


tM2A7P2 = 30086.2;
tM2A7P4 = 11808.35;












%% Mach 2.25 AOA 0

%P0
figure(1)
plot(1:size(dataM225A0, 1), dataM225A0(:, 2))

%Pi
figure(2)
plot(1:size(dataM225A0, 1), dataM225A0(:, 3))

%% Measured Pressures

lower = 1863;
upper = 2293;

[M225A0P0, M225A0Pi, M225A0P1, M225A0P2, M225A0P3, M225A0P4] = windTunnelP(dataM225A0, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M225A0M = MfromP(M225A0P0, M225A0Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 1.80984582;
ratiobot = 1.80984582;

tM225A0P1 = ratiotop*M225A0Pi
tM225A0P3 = ratiobot*M225A0Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  1.94908886
% incoming mach bot =  1.94908886
% turn angle = 20


tM225A0P2 = 12208.68;
tM225A0P4 = 12208.68;






%% Mach 2.25 AOA 3

%P0
figure(1)
plot(1:size(dataM225A3, 1), dataM225A3(:, 2))

%Pi
figure(2)
plot(1:size(dataM225A3, 1), dataM225A3(:, 3))

%% Measured Pressures

lower = 2176;
upper = 2621;

[M225A3P0, M225A3Pi, M225A3P1, M225A3P2, M225A3P3, M225A3P4] = windTunnelP(dataM225A3, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M225A3M = MfromP(M225A3P0, M225A3Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 2.13063685;
ratiobot = 1.52640716;

tM225A3P1 = ratiotop*M225A3Pi
tM225A3P3 = ratiobot*M225A3Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top = 1.82624517
% incoming mach bot = 2.06468416
% turn angle = 20


tM225A3P2 = 15765.44;
tM225A3P4 = 10092.68;




%% Mach 2.25 AOA 7

%P0
figure(1)
plot(1:size(dataM225A7, 1), dataM225A7(:, 2))

%Pi
figure(2)
plot(1:size(dataM225A7, 1), dataM225A7(:, 3))

%% Measured Pressures

lower = 2116;
upper = 2450;

[M225A7P0, M225A7Pi, M225A7P1, M225A7P2, M225A7P3, M225A7P4] = windTunnelP(dataM225A7, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M225A7M = MfromP(M225A7P0, M225A7Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 2.62251153;
ratiobot = 1.20386382;

tM225A7P1 = ratiotop*M225A7Pi
tM225A7P3 = ratiobot*M225A7Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top = 1.65119952
% incoming mach bot = 2.21341996
% turn angle = 20


tM225A7P2 = 20769.54;
tM225A7P4 = 7338.53;















%% Mach 2.5 AOA 0

%P0
figure(1)
plot(1:size(dataM25A0, 1), dataM25A0(:, 2))

%Pi
figure(2)
plot(1:size(dataM25A0, 1), dataM25A0(:, 3))

%% Measured Pressures

lower = 2251;
upper = 2585;

[M25A0P0, M25A0Pi, M25A0P1, M25A0P2, M25A0P3, M25A0P4] = windTunnelP(dataM25A0, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M25A0M = MfromP(M25A0P0, M25A0Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 1.87484823;
ratiobot = 1.87484823;

tM25A0P1 = ratiotop*M25A0Pi
tM25A0P3 = ratiobot*M25A0Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  2.11247439
% incoming mach bot =  2.11247439
% turn angle = 20


tM25A0P2 = 8489.86;
tM25A0P4 = 8489.86;




%% Mach 2.5 AOA 3

%P0
figure(1)
plot(1:size(dataM25A3, 1), dataM25A3(:, 2))

%Pi
figure(2)
plot(1:size(dataM25A3, 1), dataM25A3(:, 3))

%% Measured Pressures

lower = 1702;
upper = 2222;

[M25A3P0, M25A3Pi, M25A3P1, M25A3P2, M25A3P3, M25A3P4] = windTunnelP(dataM25A3, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M25A3M = MfromP(M25A3P0, M25A3Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop =  2.22835031;
ratiobot =  1.56756067;

tM25A3P1 = ratiotop*M25A3Pi
tM25A3P3 = ratiobot*M25A3Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  1.98772320
% incoming mach bot =  2.24012038
% turn angle = 20


tM25A3P2 = 13044.62;
tM25A3P4 = 8088.06;




%% Mach 2.75 AOA 7

%P0
figure(1)
plot(1:size(dataM25A7, 1), dataM25A7(:, 2))

%Pi
figure(2)
plot(1:size(dataM25A7, 1), dataM25A7(:, 3))

%% Measured Pressures

lower = 1500;
upper = 1970;

[M25A7P0, M25A7Pi, M25A7P1, M25A7P2, M25A7P3, M25A7P4] = windTunnelP(dataM25A7, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M25A7M = MfromP(M25A7P0, M25A7Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop =  2.77133280;
ratiobot =  1.21879344;

tM25A7P1 = ratiotop*M25A7Pi
tM25A7P3 = ratiobot*M25A7Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  1.80941963
% incoming mach bot =  2.40415576
% turn angle = 20


tM25A7P2 = 16920.3;
tM25A7P4 = 5538.08;









%% Mach 3 AOA 0

%P0
figure(1)
plot(1:size(dataM3A0, 1), dataM3A0(:, 2))

%Pi
figure(2)
plot(1:size(dataM3A0, 1), dataM3A0(:, 3))

%% Measured Pressures

lower = 1427;
upper = 1853;

[M3A0P0, M3A0Pi, M3A0P1, M3A0P2, M3A0P3, M3A0P4] = windTunnelP(dataM3A0, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M3A0M = MfromP(M3A0P0, M3A0Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop =  2.03820066;
ratiobot =  2.03820066;

tM3A0P1 = ratiotop*M3A0Pi
tM3A0P3 = ratiobot*M3A0Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  2.47207651
% incoming mach bot =  2.47207651
% turn angle = 20


tM3A0P2 = 7169.18;
tM3A0P4 = 7169.18;





%% Mach 3 AOA 3

%P0
figure(1)
plot(1:size(dataM3A3, 1), dataM3A3(:, 2))

%Pi
figure(2)
plot(1:size(dataM3A3, 1), dataM3A3(:, 3))

%% Measured Pressures

lower = 1146;
upper = 1585;

[M3A3P0, M3A3Pi, M3A3P1, M3A3P2, M3A3P3, M3A3P4] = windTunnelP(dataM3A3, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M3A3M = MfromP(M3A3P0, M3A3Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop =  2.46963356;
ratiobot =  1.66576264;

tM3A3P1 = ratiotop*M3A3Pi
tM3A3P3 = ratiobot*M3A3Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  2.32564133
% incoming mach bot =   2.61743446
% turn angle = 20


tM3A3P2 = 8893.96;
tM3A3P4 = 5117.85;




%% Mach 3 AOA 7

%P0
figure(1)
plot(1:size(dataM3A7, 1), dataM3A7(:, 2))

%Pi
figure(2)
plot(1:size(dataM3A7, 1), dataM3A7(:, 3))

%% Measured Pressures

lower = 1287;
upper = 1690;

[M3A7P0, M3A7Pi, M3A7P1, M3A7P2, M3A7P3, M3A7P4] = windTunnelP(dataM3A7, lower, upper);

%% Mach Number

% equation for online calc: sqrt((5)*(((P0/P1)^(2/7))-1))
clc
M3A7M = MfromP(M3A7P0, M3A7Pi)

%% Theoretical Leading pressure from online calc Needing Mach number and turning angle

ratiotop = 3.14550250;
ratiobot = 1.25321599;

tM3A7P1 = ratiotop*M3A7Pi
tM3A7P3 = ratiobot*M3A7Pi

%% Theoretical Leading pressure from online calc Needing Mach number, turning angle, and static pressure
% incoming mach top =  2.12745212
% incoming mach bot =  2.81494436
% turn angle = 20


tM3A7P2 = 11993.64;
tM3A7P4 = 3289.54;












%% part c M = 2 AOA = 0

tM2A0Cd = ((tM2A0P1 - tM2A0P4)*sind(10+0) + (tM2A0P3 - tM2A0P2)*sind(10-0))/(1.4*M2A0P0*(M2A0M^2)*cosd(10));
tM2A0Cl = ((tM2A0P3 - tM2A0P2)*cosd(10-0) + (tM2A0P4 - tM2A0P1)*cosd(10+0))/(1.4*M2A0P0*(M2A0M^2)*cosd(10));

M2A0Cd = ((M2A0P1 - M2A0P4)*sind(10+0) + (M2A0P3 - M2A0P2)*sind(10-0))/(1.4*M2A0P0*(M2A0M^2)*cosd(10));
M2A0Cl = ((M2A0P3 - M2A0P2)*cosd(10-0) + (M2A0P4 - M2A0P1)*cosd(10+0))/(1.4*M2A0P0*(M2A0M^2)*cosd(10));

%% part c M = 2.25 AOA = 0

tM225A0Cd = ((tM225A0P1 - tM225A0P4)*sind(10+0) + (tM225A0P3 - tM225A0P2)*sind(10-0))/(1.4*M225A0P0*(M225A0M^2)*cosd(10));
tM225A0Cl = ((tM225A0P3 - tM225A0P2)*cosd(10-0) + (tM225A0P4 - tM225A0P1)*cosd(10+0))/(1.4*M225A0P0*(M225A0M^2)*cosd(10));

M225A0Cd = ((M225A0P1 - M225A0P4)*sind(10+0) + (M225A0P3 - M225A0P2)*sind(10-0))/(1.4*M225A0P0*(M225A0M^2)*cosd(10));
M225A0Cl = ((M225A0P3 - M225A0P2)*cosd(10-0) + (M225A0P4 - M225A0P1)*cosd(10+0))/(1.4*M225A0P0*(M225A0M^2)*cosd(10));

%% part c M = 2.5 AOA = 0

tM25A0Cd = ((tM25A0P1 - tM25A0P4)*sind(10+0) + (tM25A0P3 - tM25A0P2)*sind(10-0))/(1.4*M25A0P0*(M25A0M^2)*cosd(10));
tM25A0Cl = ((tM25A0P3 - tM25A0P2)*cosd(10-0) + (tM25A0P4 - tM25A0P1)*cosd(10+0))/(1.4*M25A0P0*(M25A0M^2)*cosd(10));

M25A0Cd = ((M25A0P1 - M25A0P4)*sind(10+0) + (M25A0P3 - M25A0P2)*sind(10-0))/(1.4*M25A0P0*(M25A0M^2)*cosd(10));
M25A0Cl = ((M25A0P3 - M25A0P2)*cosd(10-0) + (M25A0P4 - M25A0P1)*cosd(10+0))/(1.4*M25A0P0*(M25A0M^2)*cosd(10));

%% part c M = 3 AOA = 0

tM3A0Cd = ((tM3A0P1 - tM3A0P4)*sind(10+0) + (tM3A0P3 - tM3A0P2)*sind(10-0))/(1.4*M3A0P0*(M3A0M^2)*cosd(10));
tM3A0Cl = ((tM3A0P3 - tM3A0P2)*cosd(10-0) + (tM3A0P4 - tM3A0P1)*cosd(10+0))/(1.4*M3A0P0*(M3A0M^2)*cosd(10));

M3A0Cd = ((M3A0P1 - M3A0P4)*sind(10+0) + (M3A0P3 - M3A0P2)*sind(10-0))/(1.4*M3A0P0*(M3A0M^2)*cosd(10));
M3A0Cl = ((M3A0P3 - M3A0P2)*cosd(10-0) + (M3A0P4 - M3A0P1)*cosd(10+0))/(1.4*M3A0P0*(M3A0M^2)*cosd(10));





%% part c M = 2 AOA = 3

tM2A3Cd = ((tM2A3P1 - tM2A3P4)*sind(10+3) + (tM2A3P3 - tM2A3P2)*sind(10-3))/(1.4*M2A3P0*(M2A3M^2)*cosd(10));
tM2A3Cl = ((tM2A3P3 - tM2A3P2)*cosd(10-3) + (tM2A3P4 - tM2A3P1)*cosd(10+3))/(1.4*M2A3P0*(M2A3M^2)*cosd(10));

M2A3Cd = ((M2A3P1 - M2A3P4)*sind(10+3) + (M2A3P3 - M2A3P2)*sind(10-3))/(1.4*M2A3P0*(M2A3M^2)*cosd(10));
M2A3Cl = ((M2A3P3 - M2A3P2)*cosd(10-3) + (M2A3P4 - M2A3P1)*cosd(10+3))/(1.4*M2A3P0*(M2A3M^2)*cosd(10));

%% part c M = 2.25 AOA = 3

tM225A3Cd = ((tM225A3P1 - tM225A3P4)*sind(10+3) + (tM225A3P3 - tM225A3P2)*sind(10-3))/(1.4*M225A3P0*(M225A3M^2)*cosd(10));
tM225A3Cl = ((tM225A3P3 - tM225A3P2)*cosd(10-3) + (tM225A3P4 - tM225A3P1)*cosd(10+3))/(1.4*M225A3P0*(M225A3M^2)*cosd(10));

M225A3Cd = ((M225A3P1 - M225A3P4)*sind(10+3) + (M225A3P3 - M225A3P2)*sind(10-3))/(1.4*M225A3P0*(M225A3M^2)*cosd(10));
M225A3Cl = ((M225A3P3 - M225A3P2)*cosd(10-3) + (M225A3P4 - M225A3P1)*cosd(10+3))/(1.4*M225A3P0*(M225A3M^2)*cosd(10));

%% part c M = 2.5 AOA = 3

tM25A3Cd = ((tM25A3P1 - tM25A3P4)*sind(10+3) + (tM25A3P3 - tM25A3P2)*sind(10-3))/(1.4*M25A3P0*(M25A3M^2)*cosd(10));
tM25A3Cl = ((tM25A3P3 - tM25A3P2)*cosd(10-3) + (tM25A3P4 - tM25A3P1)*cosd(10+3))/(1.4*M25A3P0*(M25A3M^2)*cosd(10));

M25A3Cd = ((M25A3P1 - M25A3P4)*sind(10+3) + (M25A3P3 - M25A3P2)*sind(10-3))/(1.4*M25A3P0*(M25A3M^2)*cosd(10));
M25A3Cl = ((M25A3P3 - M25A3P2)*cosd(10-3) + (M25A3P4 - M25A3P1)*cosd(10+3))/(1.4*M25A3P0*(M25A3M^2)*cosd(10));

%% part c M = 3 AOA = 3

tM3A3Cd = ((tM3A3P1 - tM3A3P4)*sind(10+3) + (tM3A3P3 - tM3A3P2)*sind(10-3))/(1.4*M3A3P0*(M3A3M^2)*cosd(10));
tM3A3Cl = ((tM3A3P3 - tM3A3P2)*cosd(10-3) + (tM3A3P4 - tM3A3P1)*cosd(10+3))/(1.4*M3A0P0*(M3A3M^2)*cosd(10));

M3A3Cd = ((M3A3P1 - M3A3P4)*sind(10+3) + (M3A3P3 - M3A3P2)*sind(10-3))/(1.4*M3A3P0*(M3A3M^2)*cosd(10));
M3A3Cl = ((M3A3P3 - M3A3P2)*cosd(10-3) + (M3A3P4 - M3A3P1)*cosd(10+3))/(1.4*M3A3P0*(M3A3M^2)*cosd(10));





%% part c M = 2 AOA = 7

tM2A7Cd = ((tM2A7P1 - tM2A7P4)*sind(10+7) + (tM2A7P3 - tM2A7P2)*sind(10-7))/(1.4*M2A7P0*(M2A7M^2)*cosd(10));
tM2A7Cl = ((tM2A7P3 - tM2A7P2)*cosd(10-7) + (tM2A7P4 - tM2A7P1)*cosd(10+7))/(1.4*M2A7P0*(M2A7M^2)*cosd(10));

M2A7Cd = ((M2A7P1 - M2A7P4)*sind(10+7) + (M2A7P3 - M2A7P2)*sind(10-7))/(1.4*M2A7P0*(M2A7M^2)*cosd(10));
M2A7Cl = ((M2A7P3 - M2A7P2)*cosd(10-7) + (M2A7P4 - M2A7P1)*cosd(10+7))/(1.4*M2A7P0*(M2A7M^2)*cosd(10));

%% part c M = 2.25 AOA = 7

tM225A7Cd = ((tM225A7P1 - tM225A7P4)*sind(10+7) + (tM225A7P3 - tM225A7P2)*sind(10-7))/(1.4*M225A7P0*(M225A7M^2)*cosd(10));
tM225A7Cl = ((tM225A7P3 - tM225A7P2)*cosd(10-7) + (tM225A7P4 - tM225A7P1)*cosd(10+7))/(1.4*M225A7P0*(M225A7M^2)*cosd(10));

M225A7Cd = ((M225A7P1 - M225A7P4)*sind(10+7) + (M225A7P3 - M225A7P2)*sind(10-7))/(1.4*M225A7P0*(M225A7M^2)*cosd(10));
M225A7Cl = ((M225A7P3 - M225A7P2)*cosd(10-7) + (M225A7P4 - M225A7P1)*cosd(10+7))/(1.4*M225A7P0*(M225A7M^2)*cosd(10));

%% part c M = 2.5 AOA = 7

tM25A7Cd = ((tM25A7P1 - tM25A7P4)*sind(10+7) + (tM25A7P3 - tM25A7P2)*sind(10-7))/(1.4*M25A7P0*(M25A7M^2)*cosd(10));
tM25A7Cl = ((tM25A7P3 - tM25A7P2)*cosd(10-7) + (tM25A7P4 - tM25A7P1)*cosd(10+7))/(1.4*M25A7P0*(M25A7M^2)*cosd(10));

M25A7Cd = ((M25A7P1 - M25A7P4)*sind(10+7) + (M25A7P3 - M25A7P2)*sind(10-7))/(1.4*M25A7P0*(M25A7M^2)*cosd(10));
M25A7Cl = ((M25A7P3 - M25A7P2)*cosd(10-7) + (M25A7P4 - M25A7P1)*cosd(10+7))/(1.4*M25A7P0*(M25A7M^2)*cosd(10));

%% part c M = 3 AOA = 7

tM3A7Cd = ((tM3A7P1 - tM3A7P4)*sind(10+7) + (tM3A7P3 - tM3A7P2)*sind(10-7))/(1.4*M3A7P0*(M3A7M^2)*cosd(10));
tM3A7Cl = ((tM3A7P3 - tM3A7P2)*cosd(10-7) + (tM3A7P4 - tM3A7P1)*cosd(10+7))/(1.4*M3A7P0*(M3A7M^2)*cosd(10));

M3A7Cd = ((M3A7P1 - M3A7P4)*sind(10+7) + (M3A7P3 - M3A7P2)*sind(10-7))/(1.4*M3A7P0*(M3A7M^2)*cosd(10));
M3A7Cl = ((M3A7P3 - M3A7P2)*cosd(10-7) + (M3A7P4 - M3A7P1)*cosd(10+7))/(1.4*M3A7P0*(M3A7M^2)*cosd(10));









%% part d Cl

% AOA 0

figure(1)
fplot(@(x) 4*(0*pi/180)/sqrt((x^2)-1), [1 3.5])
hold on
scatter(M2A0M, M2A0Cl)
scatter(M225A0M, M225A0Cl)
scatter(M25A0M, M25A0Cl)
scatter(M3A0M, M3A0Cl)
hold off
title('Supersonic linearized potential theory for a thin airfoil comparison')
legend('Thin Airfoil Theory A0A = 0','M = 2','M = 2.25','M = 2.5','M = 3','Location','southeast')
axis([1 3.5 -0.007 0.001])
xlabel('Mach Number') 
ylabel('Cl') 


%% AOA 3

figure(1)
fplot(@(x) 4*(3*pi/180)/sqrt((x^2)-1))
hold on
scatter(M2A3M, M2A3Cl)
scatter(M225A3M, M225A3Cl)
scatter(M25A3M, M25A3Cl)
scatter(M3A3M, M3A3Cl)
hold off
title('Supersonic linearized potential theory for a thin airfoil comparison')
legend('Thin Airfoil Theory AOA = 3','M = 2','M = 2.35','M = 2.5','M = 3','Location','west')
axis([1 3.5 0 0.12])
xlabel('Mach Number') 
ylabel('Cl') 

%% AOA 7

figure(1)
fplot(@(x) 4*(7*pi/180)/sqrt((x^2)-1))
hold on
scatter(M2A7M, M2A7Cl)
scatter(M225A7M, M225A7Cl)
scatter(M25A7M, M25A7Cl)
scatter(M3A7M, M3A7Cl)
hold off
title('Supersonic linearized potential theory for a thin airfoil comparison')
legend('Thin Airfoil Theory AOA = 7','M = 2','M = 2.35','M = 2.5','M = 3','Location','west')
axis([1 3.5 0 0.35])
xlabel('Mach Number') 
ylabel('Cl') 

%% Cd  AOA 0
figure(2)
fplot(@(x) (4*((0*pi/180)^2) + 2*(tand(10)^2))/sqrt((x^2)-1))
hold on
scatter(M2A0M, M2A0Cd)
scatter(M225A0M, M225A0Cd)
scatter(M25A0M, M25A0Cd)
scatter(M3A0M, M3A0Cd)
hold off
title('Supersonic linearized potential theory for a thin airfoil comparison')
legend('Thin Airfoil Theory A0A = 0','M = 2','M = 2.35','M = 2.5','M = 3','Location','northeast')
axis([1 3.5 0 0.1])
xlabel('Mach Number') 
ylabel('Cd') 

%% AOA 3
figure(2)
fplot(@(x) (4*((3*pi/180)^2) + 2*(tand(10)^2))/sqrt((x^2)-1))
hold on
scatter(M2A3M, M2A3Cd)
scatter(M225A3M, M225A3Cd)
scatter(M25A3M, M25A3Cd)
scatter(M3A3M, M3A3Cd)
hold off
title('Supersonic linearized potential theory for a thin airfoil comparison')
legend('Thin Airfoil Theory A0A = 3','M = 2','M = 2.35','M = 2.5','M = 3','Location','northeast')
axis([1 3.5 0 0.1])
xlabel('Mach Number') 
ylabel('Cd') 

%% AOA 7
figure(2)
fplot(@(x) (4*((7*pi/180)^2) + 2*(tand(10)^2))/sqrt((x^2)-1), [1.5 3.5])
hold on
scatter(M2A7M, M2A7Cd)
scatter(M225A7M, M225A7Cd)
scatter(M25A7M, M25A7Cd)
scatter(M3A7M, M3A7Cd)
hold off
title('Supersonic linearized potential theory for a thin airfoil comparison')
legend('Thin Airfoil Theory A0A = 7','M = 2','M = 2.35','M = 2.5','M = 3','Location','northeast')
axis([1 3.5 0 0.1])
xlabel('Mach Number') 
ylabel('Cd')


%% part e
%% CPx/c
% AOA 0
% M = 2
tCPxM2A0 = (0.25*tM2A0P1+0.75*tM2A0P2)/(tM2A0P1+tM2A0P2);
CPxM2A0 = (0.25*M2A0P1+0.75*M2A0P2)/(M2A0P1+M2A0P2);
% M = 2.25
tCPxM225A0 = (0.25*tM225A0P1+0.75*tM225A0P2)/(tM2A0P1+tM225A0P2);
CPxM225A0 = (0.25*M225A0P1+0.75*M225A0P2)/(M225A0P1+M225A0P2);
% M = 2.5
tCPxM25A0 = (0.25*tM25A0P1+0.75*tM25A0P2)/(tM2A0P1+tM25A0P2);
CPxM25A0 = (0.25*M25A0P1+0.75*M25A0P2)/(M225A0P1+M25A0P2);
% M = 3
tCPxM3A0 = (0.25*tM3A0P1+0.75*tM3A0P2)/(tM3A0P1+tM3A0P2);
CPxM3A0 = (0.25*M3A0P1+0.75*M3A0P2)/(M3A0P1+M3A0P2);

% AOA 3
% M = 2
tCPxM2A3 = (0.25*(tM2A3P1-tM2A3P3)+0.75*(tM2A3P2-tM2A3P4))/(tM2A3P1-tM2A3P3+tM2A3P2-tM2A3P4);
CPxM2A3 = (0.25*(M2A3P1-M2A3P3)+0.75*(M2A3P2-M2A3P4))/(M2A3P1-M2A3P3+M2A3P2-M2A3P4);
% M = 2.25
tCPxM225A3 = (0.25*(tM225A3P1-tM225A3P3)+0.75*(tM225A3P2-tM225A3P4))/(tM225A3P1-tM225A3P3+tM225A3P2-tM225A3P4);
CPxM225A3 = (0.25*(M225A3P1-M225A3P3)+0.75*(M225A3P2-M225A3P4))/(M225A3P1-M225A3P3+M225A3P2-M225A3P4);
% M = 2.75
tCPxM25A3 = (0.25*(tM25A3P1-tM25A3P3)+0.75*(tM25A3P2-tM25A3P4))/(tM25A3P1-tM25A3P3+tM25A3P2-tM25A3P4);
CPxM25A3 = (0.25*(M25A3P1-M25A3P3)+0.75*(M25A3P2-M25A3P4))/(M25A3P1-M25A3P3+M25A3P2-M25A3P4);
% M = 3
tCPxM3A3 = (0.25*(tM3A3P1-tM3A3P3)+0.75*(tM3A3P2-tM3A3P4))/(tM3A3P1-tM3A3P3+tM3A3P2-tM3A3P4);
CPxM3A3 = (0.25*(M3A3P1-M3A3P3)+0.75*(M3A3P2-M3A3P4))/(M3A3P1-M3A3P3+M3A3P2-M3A3P4);

% AOA 7
% M = 2
tCPxM2A7 = (0.25*(tM2A7P1-tM2A7P3)+0.75*(tM2A7P2-tM2A7P4))/(tM2A7P1-tM2A7P3+tM2A7P2-tM2A7P4);
CPxM2A7 = (0.25*(M2A7P1-M2A7P3)+0.75*(M2A7P2-M2A7P4))/(M2A7P1-M2A7P3+M2A7P2-M2A7P4);
% M = 2.25
tCPxM225A7 = (0.25*(tM225A7P1-tM225A7P3)+0.75*(tM225A7P2-tM225A7P4))/(tM225A7P1-tM225A7P3+tM225A7P2-tM225A7P4);
CPxM225A7 = (0.25*(M225A7P1-M225A7P3)+0.75*(M225A7P2-M225A7P4))/(M225A7P1-M225A7P3+M225A7P2-M225A7P4);
% M = 2.75
tCPxM25A7 = (0.25*(tM25A7P1-tM25A7P3)+0.75*(tM25A7P2-tM25A7P4))/(tM25A7P1-tM25A7P3+tM25A7P2-tM25A7P4);
CPxM25A7 = (0.25*(M25A7P1-M25A7P3)+0.75*(M25A7P2-M25A7P4))/(M25A7P1-M25A7P3+M25A7P2-M25A7P4);
% M = 3
tCPxM3A7 = (0.25*(tM3A7P1-tM3A7P3)+0.75*(tM3A7P2-tM3A7P4))/(tM3A7P1-tM3A7P3+tM3A7P2-tM3A7P4);
CPxM3A7 = (0.25*(M3A7P1-M3A7P3)+0.75*(M3A7P2-M3A7P4))/(M3A7P1-M3A7P3+M3A7P2-M3A7P4);

%% CPy/c
% AOA 0
% M = 2
tCPyM2A0 = (tM2A0P1-tM2A0P2-tM2A0P3+tM2A0P4)/(2*(tM2A0P1-tM2A0P2+tM2A0P3-tM2A0P4));
CPyM2A0 = (M2A0P1-M2A0P2-M2A0P3+M2A0P4)/(2*(M2A0P1-M2A0P2+M2A0P3-M2A0P4));
% M = 2.25
tCPyM225A0 = (tM225A0P1-tM225A0P2-tM225A0P3+tM225A0P4)/(2*(tM225A0P1-tM225A0P2+tM225A0P3-tM225A0P4));
CPyM225A0 = (M225A0P1-M225A0P2-M225A0P3+M225A0P4)/(2*(M225A0P1-M225A0P2+M225A0P3-M225A0P4));
% M = 2.75
tCPyM25A0 = (tM25A0P1-tM25A0P2-tM25A0P3+tM25A0P4)/(2*(tM25A0P1-tM25A0P2+tM25A0P3-tM25A0P4));
CPyM25A0 = (M25A0P1-M25A0P2-M25A0P3+M25A0P4)/(2*(M25A0P1-M25A0P2+M25A0P3-M25A0P4));
% M = 3
tCPyM3A0 = (tM3A0P1-tM3A0P2-tM3A0P3+tM3A0P4)/(2*(tM3A0P1-tM3A0P2+tM3A0P3-tM3A0P4));
CPyM3A0 = (M3A0P1-M3A0P2-M3A0P3+M3A0P4)/(2*(M3A0P1-M3A0P2+M3A0P3-M3A0P4));

% AOA 3
% M = 2
tCPyM2A3 = (tM2A3P1-tM2A3P2-tM2A3P3+tM2A3P4)/(2*(tM2A3P1-tM2A3P2+tM2A3P3-tM2A3P4));
CPyM2A3 = (M2A3P1-M2A3P2-M2A3P3+M2A3P4)/(2*(M2A3P1-M2A3P2+M2A3P3-M2A3P4));
% M = 2.25
tCPyM225A3 = (tM225A3P1-tM225A3P2-tM225A3P3+tM225A3P4)/(2*(tM225A3P1-tM225A3P2+tM225A3P3-tM225A3P4));
CPyM225A3 = (M225A3P1-M225A3P2-M225A3P3+M225A3P4)/(2*(M225A3P1-M225A3P2+M225A3P3-M225A3P4));
% M = 2.75
tCPyM25A3 = (tM25A3P1-tM25A3P2-tM25A3P3+tM25A3P4)/(2*(tM25A3P1-tM25A3P2+tM25A3P3-tM25A3P4));
CPyM25A3 = (M25A3P1-M25A3P2-M25A3P3+M25A3P4)/(2*(M25A3P1-M25A3P2+M25A3P3-M25A3P4));
% M = 3
tCPyM3A3 = (tM3A3P1-tM3A3P2-tM3A3P3+tM3A3P4)/(2*(tM3A3P1-tM3A3P2+tM3A3P3-tM3A3P4));
CPyM3A3 = (M3A3P1-M3A3P2-M3A3P3+M3A3P4)/(2*(M3A3P1-M3A3P2+M3A3P3-M3A3P4));

% AOA 7
% M = 2
tCPyM2A7 = (tM2A7P1-tM2A7P2-tM2A7P3+tM2A7P4)/(2*(tM2A7P1-tM2A7P2+tM2A7P3-tM2A7P4));
CPyM2A7 = (M2A7P1-M2A7P2-M2A7P3+M2A7P4)/(2*(M2A7P1-M2A7P2+M2A7P3-M2A7P4));
% M = 2.25
tCPyM225A7 = (tM225A7P1-tM225A7P2-tM225A7P3+tM225A7P4)/(2*(tM225A7P1-tM225A7P2+tM225A7P3-tM225A7P4));
CPyM225A7 = (M225A7P1-M225A7P2-M225A7P3+M225A7P4)/(2*(M225A7P1-M225A7P2+M225A7P3-M225A7P4));
% M = 2.75
tCPyM25A7 = (tM25A7P1-tM25A7P2-tM25A7P3+tM25A7P4)/(2*(tM25A7P1-tM25A7P2+tM25A7P3-tM25A7P4));
CPyM25A7 = (M25A7P1-M25A7P2-M25A7P3+M25A7P4)/(2*(M25A7P1-M25A7P2+M25A7P3-M25A7P4));
% M = 3
tCPyM3A7 = (tM3A7P1-tM3A7P2-tM3A7P3+tM3A7P4)/(2*(tM3A7P1-tM3A7P2+tM3A7P3-tM3A7P4));
CPyM3A7 = (M3A7P1-M3A7P2-M3A7P3+M3A7P4)/(2*(M3A7P1-M3A7P2+M3A7P3-M3A7P4));



%% part b 

M = [M2A0P0, M2A0Pi, M2A0P1, M2A0P2, M2A0P3, M2A0P4, M2A0M, tM2A0P1, tM2A0P2, tM2A0P3, tM2A0P4;
 M2A3P0, M2A3Pi, M2A3P1, M2A3P2, M2A3P3, M2A3P4, M2A3M, tM2A3P1, tM2A3P2, tM2A3P3, tM2A3P4;
 M2A7P0, M2A7Pi, M2A7P1, M2A7P2, M2A7P3, M2A7P4, M2A7M, tM2A7P1, tM2A7P2, tM2A7P3, tM2A7P4;
 M225A0P0, M225A0Pi, M225A0P1, M225A0P2, M225A0P3, M225A0P4, M225A0M, tM225A0P1, tM225A0P2, tM225A0P3, tM225A0P4;
 M225A3P0, M225A3Pi, M225A3P1, M225A3P2, M225A3P3, M225A3P4, M225A3M, tM225A3P1, tM225A3P2, tM225A3P3, tM225A3P4;
 M225A7P0, M225A7Pi, M225A7P1, M225A7P2, M225A7P3, M225A7P4, M225A7M, tM225A7P1, tM225A7P2, tM225A7P3, tM225A7P4;
 M25A0P0, M25A0Pi, M25A0P1, M25A0P2, M25A0P3, M25A0P4, M25A0M, tM25A0P1, tM25A0P2, tM25A0P3, tM25A0P4;
 M25A3P0, M25A3Pi, M25A3P1, M25A3P2, M25A3P3, M25A3P4, M25A3M, tM25A3P1, tM25A3P2, tM25A3P3, tM25A3P4;
 M25A7P0, M25A7Pi, M25A7P1, M25A7P2, M25A7P3, M25A7P4, M25A7M, tM25A7P1, tM25A7P2, tM25A7P3, tM25A7P4;
 M3A0P0, M3A0Pi, M3A0P1, M3A0P2, M3A0P3, M3A0P4, M3A0M, tM3A0P1, tM3A0P2, tM3A0P3, tM3A0P4;
 M3A3P0, M3A3Pi, M3A3P1, M3A3P2, M3A3P3, M3A3P4, M3A3M, tM3A3P1, tM3A3P2, tM3A3P3, tM3A3P4;
 M3A7P0, M3A7Pi, M3A7P1, M3A7P2, M3A7P3, M3A7P4, M3A7M, tM3A7P1, tM3A7P2, tM3A7P3, tM3A7P4];

writematrix(M,'M.xls')


%% part d: Cd vs M

x = [M2A0M, M2A3M, M2A7M, M225A0M, M225A3M, M225A7M, M25A0M, M25A3M, M25A7M, M3A0M, M3A3M, M3A7M];
ty = [tM2A0Cd, tM2A3Cd, tM2A7Cd, tM225A0Cd, tM225A3Cd, tM225A7Cd, tM25A0Cd, tM25A3Cd, tM25A7Cd, tM3A0Cd, tM25A3Cd, tM25A7Cd];
y = [M2A0Cd, M2A3Cd, M2A7Cd, M225A0Cd, M225A3Cd, M225A7Cd, M25A0Cd, M25A3Cd, M25A7Cd, M3A0Cd, M25A3Cd, M25A7Cd];

figure(1)
scatter(x, ty)
hold on
scatter(x, y)
% errorbar(x, y, errors, 'o', 'DisplayName', 'Data with Error Bars');
hold off
title('Coefficient of Drag vs Mach Number')
legend('Theoretical','Measured','Location','northeast')
xlabel('Mach Number')
ylabel('Cd')

%% Cl vs alpha

x = [0, 3, 7, 0, 3, 7, 0, 3, 7, 0, 3, 7];
ty = -1.*[tM2A0Cl, tM2A3Cl, tM2A7Cl, tM225A0Cl, tM225A3Cl, tM225A7Cl, tM25A0Cl, tM25A3Cl, tM25A7Cl, tM3A0Cl, tM25A3Cl, tM25A7Cl];
y = [M2A0Cl, M2A3Cl, M2A7Cl, M225A0Cl, M225A3Cl, M225A7Cl, M25A0Cl, M25A3Cl, M25A7Cl, M3A0Cl, M25A3Cl, M25A7Cl];

figure(1)
scatter(x, ty)
hold on
scatter(x, y)
% errorbar(x, y, errors, 'o', 'DisplayName', 'Data with Error Bars');
hold off
title('Coefficient of Lift vs Angle of Attack')
legend('Theoretical','Measured','Location','northwest')
xlabel('\alpha [degrees]')
ylabel('Cl')
axis([-0.1 7.1 -0.01 0.04])


%% Cd vs Cl

txCl = -1.*[tM2A0Cl, tM2A3Cl, tM2A7Cl, tM225A0Cl, tM225A3Cl, tM225A7Cl, tM25A0Cl, tM25A3Cl, tM25A7Cl, tM3A0Cl, tM25A3Cl, tM25A7Cl];
tyCd = [tM2A0Cd, tM2A3Cd, tM2A7Cd, tM225A0Cd, tM225A3Cd, tM225A7Cd, tM25A0Cd, tM25A3Cd, tM25A7Cd, tM3A0Cd, tM25A3Cd, tM25A7Cd];

xCl = [M2A0Cl, M2A3Cl, M2A7Cl, M225A0Cl, M225A3Cl, M225A7Cl, M25A0Cl, M25A3Cl, M25A7Cl, M3A0Cl, M25A3Cl, M25A7Cl];
yCd = [M2A0Cd, M2A3Cd, M2A7Cd, M225A0Cd, M225A3Cd, M225A7Cd, M25A0Cd, M25A3Cd, M25A7Cd, M3A0Cd, M25A3Cd, M25A7Cd];

figure(1)
scatter(txCl, tyCd)
hold on
scatter(xCl, yCd)
% errorbar(x, y, errors, 'o', 'DisplayName', 'Data with Error Bars');
hold off
title('Coefficient of Drag vs Coefficient of Lift')
legend('Theoretical','Measured','Location','northwest')
xlabel('Cl')
ylabel('Cd')
% axis([-0.1 7.1 -0.01 0.04])

%% Part e

Ms0 = [M2A0M, M225A0M, M25A0M, M3A0M];
Ms3 = [M2A3M, M225A3M, M25A3M, M3A3M];
Ms7 = [M2A7M, M225A7M, M25A7M, M3A7M];

tCpxsA0 = [tCPxM2A0, tCPxM225A0, tCPxM25A0, tCPxM3A0];
tCpxsA3 = [tCPxM2A3, tCPxM225A3, tCPxM25A3, tCPxM3A3];
tCpxsA7 = [tCPxM2A7, tCPxM225A7, tCPxM25A7, tCPxM3A7];

CpxsA0 = [CPxM2A0, CPxM225A0, CPxM25A0, CPxM3A0];
CpxsA3 = [CPxM2A3, CPxM225A3, CPxM25A3, CPxM3A3];
CpxsA7 = [CPxM2A7, CPxM225A7, CPxM25A7, CPxM3A7];


tCpysA0 = [tCPyM2A0, tCPyM225A0, tCPyM25A0, tCPyM3A0];
tCpysA3 = [tCPyM2A3, tCPyM225A3, tCPyM25A3, tCPyM3A3];
tCpysA7 = [tCPyM2A7, tCPyM225A7, tCPyM25A7, tCPyM3A7];

CpysA0 = [CPyM2A0, CPyM225A0, CPyM25A0, CPyM3A0];
CpysA3 = [CPyM2A3, CPyM225A3, CPyM25A3, CPyM3A3];
CpysA7 = [CPyM2A7, CPyM225A7, CPyM25A7, CPyM3A7];

figure(1)
scatter(Ms0, tCpxsA0)
hold on
scatter(Ms3, tCpxsA3)
scatter(Ms7, tCpxsA7)
scatter(Ms0, CpxsA0)
scatter(Ms3, CpxsA3)
scatter(Ms7, CpxsA7)
% errorbar(x, y, errors, 'o', 'DisplayName', 'Data with Error Bars');
hold off
title('Center of Pressure x-value')
legend('Theoretical AOA = 0','Theoretical AOA = 3','Theoretical AOA = 7', 'Measured AOA = 0','Measured AOA = 3','Measured AOA = 7','Location','southeast')
xlabel('Mach Number')
ylabel('Cp_x [% chord]')

figure(2)
scatter(Ms0, tCpysA0)
hold on
scatter(Ms3, tCpysA3)
scatter(Ms7, tCpysA7)
scatter(Ms0, CpysA0)
scatter(Ms3, CpysA3)
scatter(Ms7, CpysA7)
% errorbar(x, y, errors, 'o', 'DisplayName', 'Data with Error Bars');
hold off
title('Center of Pressure y-value')
legend('Theoretical AOA = 0','Theoretical AOA = 3','Theoretical AOA = 7', 'Measured AOA = 0','Measured AOA = 3','Measured AOA = 7','Location','northwest')
xlabel('Mach Number')
ylabel('Cp_y [% chord]')