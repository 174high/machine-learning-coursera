
result=[1,0,1,0,1];
m=5;

q=0;
w=0;
for i=1:5

    fprintf("result[%d]=%d",i,result(i));

    if  result(i)>=0.5
        q=q+1 
    else
        w=w+1
    end
end

fprintf(" q=%d,w=%d ",q,w);


