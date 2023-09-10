%%
N=1000:1000:100000;
ram1=24*1e-9*N.^2;
close all;
figure()
loglog(N,ram1,'r.-');
grid on;
xlabel('N: number of injected particles','interpreter','latex','FontSize',20);
ylabel('RAM in GB','interpreter','latex','FontSize',20);
t=title('RAM vs N','interpreter','latex','FontSize',20);
ax = gca;
ax.FontSize = 16; 
set(gcf, 'Color', 'k');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
set(t, 'Color', 'w');

%%
N=10000:1000:1000000;
m=10000; % Keep the m closest particles only
ram1=24*1e-9*N.^2;
ram2=12*1e-9*N.*(N+1);
ram3=32*1e-9*m.*((m+1)/2+N-10000);
ram4=16*1e-9*m.*((m+1)/2+N-10000);

figure()
loglog(N,ram1,'r.-'); hold on
loglog(N,ram2,'b.-');hold on
loglog(N,ram3,'g.-');
loglog(N,ram4,'w.-');
grid on;
xlabel('N: number of injected particles','interpreter','latex','FontSize',20);
ylabel('RAM in GB','interpreter','latex','FontSize',20);
t=title('RAM vs N','interpreter','latex','FontSize',20);
lgd=legend('Full','Remove nan','Keep 10k','float32','interpreter','latex','FontSize',20);
set(lgd, 'Location', 'northwest', 'Color', 'k','TextColor', 'w');
ax = gca;
ax.FontSize = 16; 
set(gcf, 'Color', 'k');
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w');
set(t, 'Color', 'w');
