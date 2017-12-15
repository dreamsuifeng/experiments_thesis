function PlotData(X,p,maxx, minx)
color=['b' 'g' 'r' 'c' 'm' 'y' 'k' 'w' 'b' 'g'];
y=zeros(size(X,2),1);

for i=1:size(X,2)
    [value,label]=max(p(i,:));
    y(i)=label;
end

z=unique(y);

for i=1:length(z)
    index=find(y==z(i));
    plot(X(1,index),X(2,index),strcat('.',color(z(i))),'MarkerSize',20);
    axis([minx maxx minx maxx]);
    hold on; 
end
end
