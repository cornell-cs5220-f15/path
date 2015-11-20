M = csvread('~/results.csv');

close all
hold all
orig = [];
for i=1:24
    col = M(M(:,1) == i, 2:3);
    plot(col(:,1), col(:,2))
    if (i == 1)
        orig = col;
    end
end
xlabel('n')
ylabel('Time (seconds)')

legend(cellstr(num2str((1:24)')),'Position',[0.2,0.5,0.1,0.1])

title('                        Time taken for p = 0.05, for various number of threads')

figure
hold all
for i=4:24
    col = M(M(:,1) == i, 2:3);
    plot(col(:,1), col(:,2))
end
xlabel('n')
ylabel('Time (seconds)')

legend(cellstr(num2str((4:24)')),'Position',[0.2,0.5,0.1,0.1])

title('                        Time taken for p = 0.05, for 4 to 24 threads')

figure
hold all
for i=2:8
    col = M(M(:,1) == i, 2:3);
    toPlot = zeros(length(col),1);
    j = 1;
    for j=1:length(col(:,1))
        toPlot(j,1) = orig(orig(:,1)==col(j,1),2) / col(j,2);
    end
    plot(col(:,1), toPlot)
end
ylim([0 11])
xlabel('n')
ylabel('Time (seconds)')

legend(cellstr(num2str((2:8)')),'Position',[0.2,0.5,0.1,0.1])
title('                        Speedup plot (versus serial "MPI" code): 2-8 threads')

figure
hold all
for i=9:14
    col = M(M(:,1) == i, 2:3);
    toPlot = zeros(length(col),1);
    j = 1;
    for j=1:length(col(:,1))
        toPlot(j,1) = orig(orig(:,1)==col(j,1),2) / col(j,2);
    end
    plot(col(:,1), toPlot)
end
ylim([0 11])
xlabel('n')
ylabel('Time (seconds)')

legend(cellstr(num2str((9:14)')),'Position',[0.2,0.5,0.1,0.1])
title('                        Speedup plot (versus serial "MPI" code): 9-14 threads')

figure
hold all
for i=15:20
    col = M(M(:,1) == i, 2:3);
    toPlot = zeros(length(col),1);
    j = 1;
    for j=1:length(col(:,1))
        toPlot(j,1) = orig(orig(:,1)==col(j,1),2) / col(j,2);
    end
    plot(col(:,1), toPlot)
end
ylim([0 11])
xlabel('n')
ylabel('Time (seconds)')

legend(cellstr(num2str((15:20)')),'Position',[0.2,0.5,0.1,0.1])
title('                        Speedup plot (versus serial "MPI" code): 15-20 threads')

figure
hold all
for i=21:24
    col = M(M(:,1) == i, 2:3);
    toPlot = zeros(length(col),1);
    j = 1;
    for j=1:length(col(:,1))
        toPlot(j,1) = orig(orig(:,1)==col(j,1),2) / col(j,2);
    end
    plot(col(:,1), toPlot)
end
ylim([0 11])
xlabel('n')
ylabel('Time (seconds)')

legend(cellstr(num2str((21:24)')),'Position',[0.2,0.5,0.1,0.1])
title('                        Speedup plot (versus serial "MPI" code): 21-24 threads')