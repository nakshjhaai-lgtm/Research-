pkg load io;

% Run NASA CARA's own Pc2D_Foster.m on the Alfano (2009) test cases.
% This is the reference implementation -- we are validating OUR code against it,
% not the other way round.
%
% Source: github.com/nasa/CARA_Analysis_Tools (NOSA GSC-18593-1)
%
% The input CSV is written by scripts/extract_alfano_cases.py. Columns are
% located BY NAME from the header rather than by hard-coded index, because the
% column layout changed when the covariance block was widened from 3x3 to 6x6
% and a hard-coded index silently fed position-velocity terms to the 3x3
% position covariance.

ROOT = '/home/user/CARA_Analysis_Tools/DistributedMatlab';
addpath(fullfile(ROOT, 'ProbabilityOfCollision'));
addpath(fullfile(ROOT, 'ProbabilityOfCollision', 'Utils'));
addpath(fullfile(ROOT, 'Utils'));
addpath(fullfile(ROOT, 'Utils', 'AugmentedMath'));

CSV = '/tmp/alfano_cases.csv';

% ---- parse the header and build a name -> column index map -----------------
fid = fopen(CSV, 'r');
hdr = strtrim(fgetl(fid));
fclose(fid);
names = strsplit(hdr, ',');
col = containers.Map();
for k = 1:numel(names)
    col(names{k}) = k;
end
ncols = numel(names);

% ---- read the numeric block ------------------------------------------------
T = csvread(CSV, 1, 0);
if columns(T) ~= ncols
    error('run_cara:shape', 'header has %d columns but data has %d', ncols, columns(T));
end
n = rows(T);

out = fopen('/tmp/cara_reference.csv', 'w');
fprintf(out, 'case,hbr,cara_pc2d_foster,published_pc2d\n');

printf('%4s %5s %18s %18s %10s\n', 'case', 'HBR', 'CARA Pc2D_Foster', ...
       'published', 'rel err');
printf('%s\n', repmat('-', 1, 64));

maxrel = 0;
nok = 0;
for i = 1:n
    r1 = [T(i, col('r1x')) T(i, col('r1y')) T(i, col('r1z'))];
    v1 = [T(i, col('v1x')) T(i, col('v1y')) T(i, col('v1z'))];
    r2 = [T(i, col('r2x')) T(i, col('r2y')) T(i, col('r2z'))];
    v2 = [T(i, col('v2x')) T(i, col('v2y')) T(i, col('v2z'))];

    % 3x3 position block: the six upper-triangle entries 11 12 13 22 23 33
    C1 = [T(i,col('c1_11')) T(i,col('c1_12')) T(i,col('c1_13'));
          T(i,col('c1_12')) T(i,col('c1_22')) T(i,col('c1_23'));
          T(i,col('c1_13')) T(i,col('c1_23')) T(i,col('c1_33'))];
    C2 = [T(i,col('c2_11')) T(i,col('c2_12')) T(i,col('c2_13'));
          T(i,col('c2_12')) T(i,col('c2_22')) T(i,col('c2_23'));
          T(i,col('c2_13')) T(i,col('c2_23')) T(i,col('c2_33'))];

    hbr = T(i, col('hbr'));

    % Case 12 ("LEO, identical orbits") has r_rel = v_rel = 0 exactly, so the
    % relative encounter frame y = v/|v| is 0/0 undefined and CARA's own
    % Pc2D_Foster returns NaN -- which is why CARA's published output CSV has
    % NaN for that case. Skip it rather than crash, and say so.
    vr = norm(v1 - v2);
    if vr == 0
        printf('%4d %5d %18s %18s %10s   (r_rel=v_rel=0: encounter frame undefined)\n', ...
               T(i, col('case')), hbr, 'skip', 'NaN', '-');
        fprintf(out, '%d,%g,NaN,%.17g\n', T(i, col('case')), hbr, NaN);
        continue;
    end

    Pc = Pc2D_Foster(r1, v1, C1, r2, v2, C2, hbr, 1e-8, 'circle');

    ref = T(i, col('published_pc2d'));
    if isnan(ref) || isnan(Pc)
        printf('%4d %5d %18s %18s %10s\n', T(i, col('case')), hbr, ...
               '(NaN)', '(NaN)', '-');
    else
        rel = abs(Pc - ref) / ref;
        maxrel = max(maxrel, rel);
        nok = nok + 1;
        printf('%4d %5d %18.10e %18.10e %10.2e\n', ...
               T(i, col('case')), hbr, Pc, ref, rel);
    end
    fprintf(out, '%d,%g,%.17g,%.17g\n', T(i, col('case')), hbr, Pc, ref);
end
fclose(out);

printf('%s\n', repmat('-', 1, 64));
printf('cases compared: %d\n', nok);
printf('worst rel dev of CARA code vs its published values: %.3e\n', maxrel);
printf('CARA unit-test RelTol is 1e-3.\n');
if maxrel > 1e-6
    printf('*** FAIL: CARA is not reproducing its own published values ***\n');
    error('CARA self-check failed');
end

