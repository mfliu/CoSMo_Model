clear all;
close all;
clc;

load trajectories_final.mat;
load s1_data.mat;

idx_range = 1:50;
train_idx = randperm(50, 40);
test_idx = setdiff(idx_range, train_idx);

test_traj = trajectories(test_idx, 1:132, :);
mean_traj = squeeze(mean(trajectories(train_idx, 1:132, :), 1));

s1_range = 1:237;
s1_train_idx = randperm(237, 40);
s1_test_idx_pre = setdiff(s1_range, s1_train_idx);
s1_test_idx = randperm(length(s1_test_idx_pre), 10);
s1_test_idx = s1_test_idx_pre(s1_test_idx);
s1_cca = squeeze(mean(sL(:, s1_train_idx, :), 2))';
s1_test = sL(:, s1_test_idx, :);

trajectories_cca = mean_traj(:, 1:7);

joint_vel = mean_traj(:, 24:30);
joint_vel_test = test_traj(:, :, 24:30);

%end_x_diff = [1; diff(mean_traj(:, 47), 1)];
%end_y_diff = [1; diff(mean_traj(:, 48), 1)];
%end_z_diff = [1; diff(mean_traj(:, 49), 1)];
%jac = zeros(132, 23*3);
%for i = 1:23
%    joint_diff = [0; diff(mean_traj(:, i), 1)];
%    jac(:, i) = joint_diff ./ end_x_diff;
%    jac(:,i+1) = joint_diff ./ end_y_diff;
%    jac(:, i+2) = joint_diff ./ end_z_diff;
%end

%traj_acc = [zeros(1, 7); diff(mean_traj(:, 24:30),1)];


time_lag =  10;
del_traj = zeros(132-time_lag, 7);
del_traj_test = zeros(length(test_idx), 132-time_lag, 7);
for i = 1:132-time_lag
    del_traj(i, :) = mean_traj(i+time_lag, 1:7) - mean_traj(i,1:7);
end
for i = 1:length(test_idx)
    for j = 1:132-time_lag
        del_traj_test(i, j, :) = test_traj(i, j+time_lag, 1:7) - test_traj(i, j, 1:7);
    end
end
s1_del_test = s1_test(:, :, 1:132-time_lag);

[A_joint_angles, B_joint_angles, r_joint_angles, U_joint_angles, V_joint_angles, stats_joint_angles] = canoncorr(trajectories_cca, s1_cca);
[A_joint_vel, B_joint_vel, r_joint_vel, U_joint_vel, V_joint_vel, stats_joint_vel] = canoncorr(joint_vel, s1_cca);
%[A_jac, B_jac, r_jac, U_jac, V_jac, stats_jac] = canoncorr(jac, s1_cca);
%[A_acc, B_acc, r_acc, U_acc, V_acc, stats_acc] = canoncorr(traj_acc, s1_cca);
[A_del, B_del, r_del, U_del, V_del, stats_del] = canoncorr(del_traj, s1_cca(1:132-time_lag, :));

%r_test_angles = zeros(1, length(test_idx));
figure; hold on;
%for i = 1:length(test_idx)
    U_joint_test = squeeze(mean(test_traj(:, :, 1:7), 1)) * A_joint_angles;
    V_joint_test = squeeze(mean(s1_test(:, :, :), 2))' * B_joint_angles;
    r = corrcoef(U_joint_test(:, 1), V_joint_test(:, 1));
    r_test_angles = r(1,2)^2;
    plot(U_joint_test(:, 1), 'b');
    plot(V_joint_test(:, 1), 'g');
%end

%r_test_vel = zeros(1, length(test_idx));
figure; hold on;
%for i = 1:length(test_idx)
    U_vel_test = squeeze(mean(joint_vel_test(:, :, :), 1)) * A_joint_vel;
    V_vel_test = squeeze(mean(s1_test(:, :, :), 2))' * B_joint_vel;
    r = corrcoef(U_vel_test(:, 1), V_vel_test(:, 1));
    r_test_vel = r(1,2)^2;
    plot(U_vel_test(:, 1), 'b');
    plot(V_vel_test(:, 1), 'g');
%end



%r_test_del = zeros(1, length(test_idx));
figure; hold on;
%for i = 1:length(test_idx)
    U_del_test = squeeze(mean(del_traj_test(:, :, :), 1)) * A_del;
    V_del_test = squeeze(mean(s1_del_test(:, :, :), 2))' * B_del;
    r = corrcoef(U_del_test(:, 1), V_del_test(:, 1));
    r_test_del = r(1,2)^2;
    plot(U_del_test(:, 1), 'b');
    plot(V_del_test(:, 1), 'g');
%end

