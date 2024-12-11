%% M = 2, run 1

A = readmatrix('M2_1.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat2r1 = mean(mean(imcrop(Irot,rect)));
dIrat2r1 = mean(std(imcrop(Irot,rect)));

%% M = 2, run 2

A = readmatrix('M2_2.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat2r2 = mean(mean(imcrop(Irot,rect)));
dIrat2r2 = mean(std(imcrop(Irot,rect)));

%% M = 2, run 3

A = readmatrix('M2_3.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat2r3 = mean(mean(imcrop(Irot,rect)));
dIrat2r3 = mean(std(imcrop(Irot,rect)));

%% M = 2.5, run 1

A = readmatrix('M2.5_1.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat25r1 = mean(mean(imcrop(Irot,rect)));
dIrat25r1 = mean(std(imcrop(Irot,rect)));

%% M = 2.5, run 2

A = readmatrix('M2.5_2.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat25r2 = mean(mean(imcrop(Irot,rect)));
dIrat25r2 = mean(std(imcrop(Irot,rect)));

%% M = 2.5, run 3

A = readmatrix('M2.5_3.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat25r3 = mean(mean(imcrop(Irot,rect)));
dIrat25r3 = mean(std(imcrop(Irot,rect)));

%% M = 3, run 1

A = readmatrix('M3_1.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat3r1 = mean(mean(imcrop(Irot,rect)));
dIrat3r1 = mean(std(imcrop(Irot,rect)));

%% M = 3, run 2

A = readmatrix('M3_2.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat3r2 = mean(mean(imcrop(Irot,rect)));
dIrat3r2 = mean(std(imcrop(Irot,rect)));

%% M = 3, run 3

A = readmatrix('M3_3.dat');

B = zeros(600,800);

counter = 1;
x = 1;
while x<801
    y = 1;
    while y<601
        B(y,x) = A(counter,3);
        counter = counter+1;
        y = y+1;
    end
    x = x+1;
end


Irot = imrotate(B,-90, 'crop');
imshow(Irot)
rect = getrect;

Irat3r3 = mean(mean(imcrop(Irot,rect)));
dIrat3r3 = mean(std(imcrop(Irot,rect)));

%%
Irat = [Irat2r1, Irat2r2, Irat2r3, Irat25r1, Irat25r2, Irat25r3, Irat3r1, Irat3r2, Irat3r3];
dIrat = [dIrat2r1, dIrat2r2, dIrat2r3, dIrat25r1, dIrat25r2, dIrat25r3, dIrat3r1, dIrat3r2, dIrat3r3];
