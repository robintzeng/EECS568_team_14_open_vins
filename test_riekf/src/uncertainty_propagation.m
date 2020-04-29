syms gx R vx px dx Tr Tv Tp Tba Tbg Td dt

assume(dt, 'real')

% For state X = [R v p b_a b_g d]
% For uncertainty T = [Tr Tv Tp Tba Tbg Td]

% Phi_dot = A * Phi
A = sym(zeros(6));
A(2, 1) = gx;
A(1, 5) = -R;
A(2, 5) = -vx*R;
A(3, 5) = -px*R;
A(4, 5) = -dx*R;
A(2, 6) = -R;

uncertainty = sym(eye(6));
uncertainty(1, :) = [Tr, Tv, Tp, Tba, Tbg, Td];

Phi = eye(size(A)) + A*dt; % Approximation of expm(A) = I + A*dt
P_old = eye(6);

% Uncertainty Propagation
P_new = Phi * P_old * Phi'

% latexit(P_new)