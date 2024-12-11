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
rect = getrect

Irat = mean(mean(imcrop(Irot,rect)))
dIrat = mean(std(imcrop(Irot,rect)))