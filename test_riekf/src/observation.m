% Landmark true position
m = [5; 3; 0; 1]; % homogeneous coordinates

% Robot starts at [0, 0, 0] with R = I_3
robostate = eye(4);

for i=1:5    
    motion = zeros(4);
    motion(1, end) = 0.5;
    robostate = robostate + motion;
    inv_robostate = inv(robostate);
    PI = get_PI(inv_robostate(1:3, 1:3), -inv_robostate(1:3, end));
    disp(PI*m);
end

% scatter(p(0), p(1))


function PI = get_PI(R, p)
    focal_length = 1;
    C = focal_length * eye(4);
    C = C(1:end-1, 1:end);
    PI = C * [[R; zeros(1, 3)], [p; 1]];
end