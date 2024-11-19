% theoretical pressures M = 2
tP1M2A0 = 0;
tP2M2A0 = 0;
tP3M2A0 = 0;
tP4M2A0 = 0;

tP1M2A3 = 0;
tP2M2A3 = 0;
tP3M2A3 = 0;
tP4M2A3 = 0;

tP1M2A7 = 0;
tP2M2A7 = 0;
tP3M2A7 = 0;
tP4M2A7 = 0;

% theoretical pressures M = 2.25
tP1M225A0 = 0;
tP2M225A0 = 0;
tP3M225A0 = 0;
tP4M225A0 = 0;

tP1M225A3 = 0;
tP2M225A3 = 0;
tP3M225A3 = 0;
tP4M225A3 = 0;

tP1M225A7 = 0;
tP2M225A7 = 0;
tP3M225A7 = 0;
tP4M225A7 = 0;

% theoretical pressures M = 2.75
tP1M275A0 = 0;
tP2M275A0 = 0;
tP3M275A0 = 0;
tP4M275A0 = 0;

tP1M275A3 = 0;
tP2M275A3 = 0;
tP3M275A3 = 0;
tP4M275A3 = 0;

tP1M275A7 = 0;
tP2M275A7 = 0;
tP3M275A7 = 0;
tP4M275A7 = 0;

% theoretical pressures M = 3
tP1M3A0 = 0;
tP2M3A0 = 0;
tP3M3A0 = 0;
tP4M3A0 = 0;

tP1M3A3 = 0;
tP2M3A3 = 0;
tP3M3A3 = 0;
tP4M3A3 = 0;

tP1M3A7 = 0;
tP2M3A7 = 0;
tP3M3A7 = 0;
tP4M3A7 = 0;

% measured pressures M = 2
P1M2A0 = 0;
P2M2A0 = 0;
P3M2A0 = 0;
P4M2A0 = 0;

P1M2A3 = 0;
P2M2A3 = 0;
P3M2A3 = 0;
P4M2A3 = 0;

P1M2A7 = 0;
P2M2A7 = 0;
P3M2A7 = 0;
P4M2A7 = 0;

% measured pressures M = 2.25
P1M225A0 = 0;
P2M225A0 = 0;
P3M225A0 = 0;
P4M225A0 = 0;

P1M225A3 = 0;
P2M225A3 = 0;
P3M225A3 = 0;
P4M225A3 = 0;

P1M225A7 = 0;
P2M225A7 = 0;
P3M225A7 = 0;
P4M225A7 = 0;

% measured pressures M = 2.75
P1M275A0 = 0;
P2M275A0 = 0;
P3M275A0 = 0;
P4M275A0 = 0;

P1M275A3 = 0;
P2M275A3 = 0;
P3M275A3 = 0;
P4M275A3 = 0;

P1M275A7 = 0;
P2M275A7 = 0;
P3M275A7 = 0;
P4M275A7 = 0;

% measured pressures M = 3
P1M3A0 = 0;
P2M3A0 = 0;
P3M3A0 = 0;
P4M3A0 = 0;

P1M3A3 = 0;
P2M3A3 = 0;
P3M3A3 = 0;
P4M3A3 = 0;

P1M3A7 = 0;
P2M3A7 = 0;
P3M3A7 = 0;
P4M3A7 = 0;

%% part c
% AOA = 0
Cd = ((P4 - P1)*sind(10-0) + (P3 - P2)*sind(10-0))/(1.4*P0*(M^2)*cosd(10));
Cl = ((P3 - P2)*cosd(10-0) + (P4 - P1)*cosd(10+0))/(1.4*P0*(M^2)*cosd(10));



%% part d

% Cl
% AOA 3

x = 1.5:0.001:3.5;
y = 0.1.*ones(size(x));

figure(1)
fplot(@(x) 4*(3*pi/180)/sqrt((x^2)-1), [1.5 3.5])
hold on
plot(x, y)
hold off

%% AOA 7
figure(2)
fplot(@(x) 4*(7*pi/180)/sqrt((x^2)-1), [1.5 3.5])

% Cd
%% AOA 0
figure(2)
fplot(@(x) (4*((0*pi/180)^2) + 2*(tand(10)^2))/sqrt((x^2)-1), [1.5 3.5])

%% AOA 3
figure(2)
fplot(@(x) (4*((3*pi/180)^2) + 2*(tand(10)^2))/sqrt((x^2)-1), [1.5 3.5])

%% AOA 7
figure(2)
fplot(@(x) (4*((7*pi/180)^2) + 2*(tand(10)^2))/sqrt((x^2)-1), [1.5 3.5])

% part e

%% CPx/c
% AOA 0
% M = 2
tCPxcA0M2 = (0.25*tP1+0.75*tP2)/(tP1+tP2);
CPxcA0M2 = (0.25*P1+0.75*P2)/(P1+P2);
% M = 2.25

% M = 2.75

% M = 3


% AOA 3
% M = 2
tCPxcA3M2 = (0.25*(tP1-tP3)+0.75*(tP2-tP4))/(tP1-tP3+tP2-tP4);
% M = 2.25

% M = 2.75

% M = 3

% AOA 7
% M = 2
tCPxcA7M2 = (0.25*(tP1-tP3)+0.75*(tP2-tP4))/(tP1-tP3+tP2-tP4);
% M = 2.25

% M = 2.75

% M = 3

%% CPy/c
% AOA 0
% M = 2
tCPycA0M2 = (tP1-tP2-tP3+tP4)/(2*(tP1-tP2+tP3-tP4));
% M = 2.25

% M = 2.75

% M = 3


% AOA 3
% M = 2
tCPycA3M2 = (tP1-tP2-tP3+tP4)/(2*(tP1-tP2+tP3-tP4));
% M = 2.25

% M = 2.75

% M = 3

% AOA 7
% M = 2
tCPycA7M2 = (tP1-tP2-tP3+tP4)/(2*(tP1-tP2+tP3-tP4));
% M = 2.25

% M = 2.75

% M = 3