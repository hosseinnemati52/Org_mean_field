import numpy as np
import matplotlib.pyplot as plt



# Define two linear relationships for "C" and "WT" (adjust slopes as needed)


params = np.loadtxt('parameters_errors.csv', delimiter=',')

S =   params[0,0]
v_S = params[1,0]
mu_W = params[2,0]
mu_C = params[3,0]
a_W = params[4,0]
a_C = params[5,0]
r_D = params[6,0]

# Example x-range for [GF]
alpha = np.linspace(0, 1.1*a_W, 200)

v_G1_W = mu_W * alpha
v_G1_C = mu_C * alpha



# Compute corresponding uptake rates on each line
v_G1_W_eq = mu_W * a_W
v_G1_C_eq = mu_C * a_C

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the lines
ax.plot(alpha, v_G1_W, label='WT', color='m')
ax.plot(alpha, v_G1_C, label='C', color='g')


# Add dashed lines for gf_C
ax.axvline(a_W, color='gray', linestyle='--', alpha=0.7, ymax=0.226)
ax.axhline(v_G1_W_eq , color='gray', linestyle='--', alpha=0.7, xmax=0.85)


ax.axvline(a_C, color='gray', linestyle='--', alpha=0.7, ymax=0.31)
ax.axhline(v_G1_C_eq , color='gray', linestyle='--', alpha=0.7, xmax=0.32)

a_mid = 0.5*a_C+0.5*a_W
ax.axvline(a_mid, color='b', linestyle='--', alpha=0.7, ymax=0.6)
ax.axhline(a_mid * mu_C, color='b', linestyle='--', alpha=0.7, xmax=0.6)
ax.axhline(a_mid * mu_W, color='b', linestyle='--', alpha=0.7, xmax=0.6)

plt.scatter([a_W], [v_G1_W_eq], marker='o', color='m', s=70, zorder=10)
plt.scatter([a_C], [v_G1_C_eq], marker='o', color='g', s=70, zorder=10)
plt.scatter([a_mid], [a_mid * mu_W], marker='s', color='m', s=70, facecolors='None', zorder=10)
plt.scatter([a_mid], [a_mid * mu_C], marker='s', color='g', s=70, facecolors='None', zorder=10)

# # Add dashed lines for gf_W
# ax.axvline(gf_C, color='gray', linestyle='--', alpha=0.7, ymax=0.24)
# ax.axvline(gf_W, color='gray', linestyle='--', alpha=0.7, ymax=0.7)
# ax.axhline(uptake_WT_at_gfW, color='gray', linestyle='--', alpha=0.7, xmax=0.7)


# ax.axhline(5 * slope_C, color='gray', linestyle='--', alpha=0.7, xmax=0.5)

# Annotate the intersection points
# ax.text(gf_C + 0.2, uptake_C_at_gfC, 
#         f'({gf_C}, {uptake_C_at_gfC:.1f})', 
#         fontsize=9, color='blue')

# ax.text(gf_W + 0.2, uptake_WT_at_gfW, 
#         f'({gf_W}, {uptake_WT_at_gfW:.1f})', 
#         fontsize=9, color='orange')

# Label axes
ax.set_xlabel(r'$\tilde{\alpha}$', fontsize=20)
# ax.set_ylabel('Uptake Rate', fontsize=20)
ax.set_ylabel(r'$\tilde{v}_{\mathrm{G1}}$', fontsize=20)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel(fontsize=20)
# plt.ylabel(fontsize=20)

# Add a legend
ax.legend(fontsize=20)
plt.gca().set_aspect(0.2)
# # Set a reasonable limit for clarity
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 40])  # Adjust if needed

# plt.axis('equal')

plt.tight_layout()
# plt.savefig('first.PNG', dpi=300)
plt.savefig('v_G1_plot.PNG', dpi=400)
plt.show()


