clear all

% This makes the list of mode parameters uwlm

%% fibre parameters
a = 12.5; %12.5; % core radius (um) Thorlabs fibre is 25um core radius % optical inverter paper fig 1: a = 12.5, NA = 0.1, UWLM2PM2Pix width = 31.
a_std = a*1e-6;
NA = 0.1; %0.22; % NA Thorlabs fibre is 0.22NA
n_c = 1.4570; % refractive index of core
n_out = (n_c^2 - NA^2)^0.5;
%n_clad = ; % refractive index of cladding
%NA = (n_core^2 - n_clad^2)^0.5;

%% wavelength
lam = 0.633; %0.633; %1.064; % 0.633;
%lam = 0.633; % wavelength (um)
lam_std = lam*1e-6;
k = 2*pi/lam;
k_std = 2*pi/lam_std;
v = a*k*NA; % vnumber (i.e. normalised frequency) of fibre
k_stdn_c = k_std*n_c;

%% numerically solve dispersion relation
u=0:(k*a*NA)/10000:a*k*NA; % range in which to look for parameter u, you might need finer step than k*a*NA/1000 depending on
% your fibre parameters!!!!

for l=0:70%500
    z=real(u.*besselj(l-1,u)./besselj(l,u)+sqrt(v.^2-u.^2).*besselk(l-1,sqrt(v.^2-u.^2))./besselk(l,sqrt(v.^2-u.^2))); % scalar characteristic equation
    %plot(z)
    ind=find(max(-diff(sign(z))/2,0)); % approximate position of roots of the characteristic equation
    %plot(z)
    % this for-loop very accurately finds all the roots of the characteristic equation for given
    % l parameter
    for m=1:length(ind)
        ul(l+1,m)=abs(fminsearch(@(ux) abs(abs(ux).*besselj(l-1,abs(ux))./besselj(l,abs(ux))+sqrt(v.^2-abs(ux).^2).*besselk(l-1,sqrt(v.^2-abs(ux).^2))./besselk(l,sqrt(v.^2-abs(ux).^2))).^2,u(ind(m)-1)));
        wl(l+1,m)=sqrt(v.^2-ul(l+1,m).^2);
    end
end

% re-ordering of fibre modes so that 
ind=1;
for l=-size(ul,1)+1:size(ul,1)-1
    for m=0:size(ul,2)-1
        if ul(abs(l)+1,m+1)~=0
            uwlm(ind,:)=[ul(abs(l)+1,m+1),wl(abs(l)+1,m+1),l,m];
            ind=ind+1;
        end
    end
end

[N_modes, ~] = size(uwlm);

uwlmWithParams(1,:) = [a,NA,n_c,lam];
uwlmWithParams(2:N_modes+1,:) = uwlm;
dlmwrite('uwlm.txt',uwlmWithParams);
