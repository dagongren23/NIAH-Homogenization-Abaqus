"""
NIAH-ABAQUS: Visualization utility function
Author: Zhihui Liu
Date:   2026/04/07
Email:  dutme_lzh@163.com
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit
from matplotlib.ticker import MaxNLocator, ScalarFormatter

def _Te_Ts(theta):
    m = np.cos(theta)
    n = np.sin(theta)

    Te = np.array([
        [ m*m,  n*n,  m*n],
        [ n*n,  m*m, -m*n],
        [-2*m*n, 2*m*n, m*m - n*n]
    ], dtype=float)

    Ts = np.array([
        [ m*m,  n*n,  2*m*n],
        [ n*n,  m*m, -2*m*n],
        [-m*n,  m*n,  m*m - n*n]
    ], dtype=float)
    return Te, Ts

def _rotate_Q(Q, theta):
    Te, Ts = _Te_Ts(theta)
    return np.linalg.solve(Ts, Q.dot(Te))  # Ts^{-1} * Q * Te

def _membrane_constants_from_A(A, h):
    a = np.linalg.inv(A)  # eps0 = a * N
    # N = sigma*h => eps = a*(sigma*h) => sigma/eps = 1/(a*h)
    Ex  = 1.0 / (a[0,0] * h)
    Ey  = 1.0 / (a[1,1] * h)
    Gxy = 1.0 / (a[2,2] * h)
    nuxy = -a[0,1] / a[0,0]
    nuyx = -a[0,1] / a[1,1]
    return Ex, Ey, Gxy, nuxy, nuyx

def _bending_constants_from_D(D, h):
    d = np.linalg.inv(D)  # kappa = d * M
    Exb  = 12.0 / (d[0,0] * h**3)
    Eyb  = 12.0 / (d[1,1] * h**3)
    Gxyb = 12.0 / (d[2,2] * h**3)
    nuxy_b = -d[0,1] / d[0,0]
    nuyx_b = -d[0,1] / d[1,1]
    return Exb, Eyb, Gxyb, nuxy_b, nuyx_b

def _filtered_rticks(rmin, rmid, rmax, fmt='%.1e', rel_tol=0.18, min_rel_sep=0.10,
                     fallback='mean'):
    ticks = np.array([rmin, rmid, rmax], dtype=float)
    rng = rmax - rmin

    if rng <= 0 or np.isclose(rng, 0.0):
        return [float(rmid)]

    if rng / max(abs(rmax), 1.0) < min_rel_sep:
        return [float(np.mean(ticks))] if fallback == 'mean' else [float(rmid)]

    kept = [ticks[0]]
    kept_labels = [fmt % ticks[0]]

    for t in ticks[1:]:
        label = fmt % t
        far_enough = abs(t - kept[-1]) >= rel_tol * rng
        label_new = label != kept_labels[-1]
        if far_enough and label_new:
            kept.append(t)
            kept_labels.append(label)

    if len(kept) <= 1:
        return [float(np.mean(ticks))] if fallback == 'mean' else [float(rmid)]

    return kept

def visual_plate_polar(EH, h, ntheta=361, title_prefix=""):
    """
    Plot directional plate properties in polar coordinates:
      - equivalent membrane modulus Ex_mem(theta)
      - equivalent bending modulus Ex_ben(theta)
    """
    A = EH[0:3, 0:3]
    D = EH[3:6, 3:6]

    thetas = np.linspace(0.0, 2 * np.pi, ntheta)
    Ex_mem = np.zeros_like(thetas)
    G_mem = np.zeros_like(thetas)
    nu_mem = np.zeros_like(thetas)

    Ex_ben = np.zeros_like(thetas)
    G_ben = np.zeros_like(thetas)
    nu_ben = np.zeros_like(thetas)

    for k, th in enumerate(thetas):
        Arot = _rotate_Q(A, th)
        Drot = _rotate_Q(D, th)

        ex, ey, g, nuxy, nuyx = _membrane_constants_from_A(Arot, h)
        Ex_mem[k] = ex
        G_mem[k] = g
        nu_mem[k] = nuxy

        exb, eyb, gb, nuxy_b, nuyx_b = _bending_constants_from_D(Drot, h)
        Ex_ben[k] = exb
        G_ben[k] = gb
        nu_ben[k] = nuxy_b

    # Publication-oriented polar-plot settings.
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Configure consistent typography.
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['figure.dpi'] = 600

    fig = plt.figure(figsize=(6.8, 3.8), dpi=600, constrained_layout=True)

    color_mem = plt.cm.viridis(0.72)
    color_ben = plt.cm.coolwarm(0.18)

    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122, projection='polar')

    ax1.plot(
        thetas, Ex_mem,
        color=color_mem,
        lw=2.0,
        antialiased=True,
        solid_joinstyle='round',
        solid_capstyle='round'
    )

    ax2.plot(
        thetas, Ex_ben,
        color=color_ben,
        lw=2.0,
        antialiased=True,
        solid_joinstyle='round',
        solid_capstyle='round'
    )

    # Use LaTeX-style titles with consistent weight.
    ax1.set_title(r"$\mathbf{Membrane\ E_x(\theta)}$", fontsize=18, y=1.16, fontweight='bold')
    ax2.set_title(r"$\mathbf{Bending\ E_x^{\mathrm{ben}}(\theta)}$", fontsize=18, y=1.16, fontweight='bold')

    for ax, data in zip([ax1, ax2], [Ex_mem, Ex_ben]):
        rmin = np.min(data)
        rmax = np.max(data)
        rmid = 0.5 * (rmin + rmax)

        # Retain one radial reference value for an uncluttered plot.
        ax.set_rticks([rmax])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        # Configure tick labels explicitly for Matplotlib compatibility.
        ax.tick_params(axis='both', labelsize=14, pad=2)

        ax.set_thetagrids(
            angles=[0, 90, 180, 270],
            labels=[r'$0^\circ$', r'$90^\circ$', r'$180^\circ$', r'$270^\circ$'],
            fontsize=18
        )

        # Apply the requested weight to angular tick labels.
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        # Apply the requested weight to radial tick labels.
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.spines['polar'].set_linewidth(1.2)
        ax.set_rlabel_position(67.5)
        ax.set_facecolor('white')

    return fig, (thetas, Ex_mem, Ex_ben, G_mem, G_ben, nu_mem, nu_ben)

@jit(nopython=True)
def generate(homogenized_constitutive_matrix):
    C = np.full((3, 3, 3, 3), 0.)
    for i in range(6):
        for j in range(6):
            (a, b) = change(i)
            (c, d) = change(j)
            C[a, b, c, d] = homogenized_constitutive_matrix[i, j]
    for i in range(3):
        if i == 2:
            j = 0
        else:
            j = i + 1
        for m in range(3):
            if m == 2:
                n = 0
            else:
                n = m + 1
            C[j, i, n, m] = C[i, j, m, n]
            C[j, i, m, n] = C[i, j, m, n]
            C[i, j, n, m] = C[i, j, m, n]
            C[j, i, m, m] = C[i, j, m, m]
            C[m, m, j, i] = C[m, m, i, j]
    return C

@jit(nopython=True)
def change(w):
    """change the index 4 5 6 to 23 31 12"""
    if w < 3:
        a = w
        b = w
    elif w == 3:
        a = 1
        b = 2
    elif w == 4:
        a = 2
        b = 0
    elif w == 5:
        a = 0
        b = 1
    return a, b

@jit(nopython=True)
def ToMatrix(C):
    """
    Convert the fourth-order tensor into a 6x6 CH constitutive matrix
    """
    CH = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            a, b = change(i)
            c, d = change(j)
            CH[i, j] = C[a, b, c, d]
    return CH

def modulus(homogenized_constitutive_matrix):
    """
    Used for calculating the elastic modulus constant based on CH
    """
    E = np.zeros((6, 1))
    try:
        S = np.linalg.inv(homogenized_constitutive_matrix)
    except np.linalg.LinAlgError:
        return E
    E[0] = 1 / S[0, 0]
    E[1] = 1 / S[1, 1]
    E[2] = 1 / S[2, 2]
    E[3] = 1 / S[3, 3]
    E[4] = 1 / S[4, 4]
    E[5] = 1 / S[5, 5]
    return E

def visual(CH):
    """
    By undergoing rotational changes, the elastic modulus in various directions is calculated
    """
    tensor = generate(CH)
    a = np.linspace(0, 2 * np.pi, num=150)
    e = np.linspace(-np.pi / 2, np.pi / 2, num=150)
    a, e = np.meshgrid(a, e)
    E1 = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            trans_z = np.array(
                [[np.cos(a[i][j]), -np.sin(a[i][j]), 0.], [np.sin(a[i][j]), np.cos(a[i][j]), 0.], [0., 0., 1.]])
            trans_y = np.array(
                [[np.cos(e[i][j]), 0., np.sin(e[i][j])], [0., 1., 0.], [-np.sin(e[i][j]), 0., np.cos(e[i][j])]])
            N_tensor = transform(tensor, np.matmul(trans_y, trans_z))
            N_CH = ToMatrix(N_tensor)
            E = modulus(N_CH)
            E1[i, j] = E[0].item()  # scalar

    x, y, z = sph2cart(a, e, E1)
    # Map the directional surface back from the Abaqus preprocessing rotation.
    periodicity_ch = 3
    if periodicity_ch == 2:
        # Inverse of a +90-degree rotation about x.
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]], dtype=float)
    elif periodicity_ch == 1:
        # Inverse of a +90-degree rotation about y.
        R = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0]], dtype=float)
    else:
        # no
        R = np.eye(3, dtype=float)

    X = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
    Y = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
    Z = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z
    x, y, z = X, Y, Z

    V = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    V_normalized = (V - np.min(V)) / (np.max(V) - np.min(V))
    # Publication-oriented 3D-plot settings.
    import matplotlib as mpl
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    import matplotlib.cm as cm

    # 1. Global typography and line widths.
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.labelweight": "bold",
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(6.0, 5.0), dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet(V_normalized), linewidth=0, antialiased=False,
                    edgecolor='none', rstride=1, cstride=1)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(V_normalized)
    ax.view_init(elev=25, azim=-125)
    fig = plt.figure(figsize=(6.5, 5.5), dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    # 2. Plot the directional-modulus surface.
    cmap = plt.get_cmap('jet')
    ax.plot_surface(x, y, z, facecolors=cmap(V_normalized), linewidth=0,
                    antialiased=False, edgecolor='none', rstride=1, cstride=1)

    # 3. Annotate the colorbar with the physical modulus values.
    m = cm.ScalarMappable(cmap=cmap)
    # Use the physical modulus values; normalized values are only for colors.
    m.set_array(V)
    cbar = fig.colorbar(m, ax=ax, shrink=0.75, aspect=12, pad=0.08)
    cbar.set_label("Elastic Modulus", fontweight='bold', labelpad=10, fontsize=18)
    cbar.ax.tick_params(direction='out', width=1.0)

    # Format the colorbar in scientific notation when appropriate.
    cbar_formatter = ScalarFormatter(useMathText=True)
    cbar_formatter.set_scientific(True)
    cbar_formatter.set_powerlimits((-2, 3))
    cbar.ax.yaxis.set_major_formatter(cbar_formatter)

    # 4. Lock equal axis ranges and aspect ratios.
    all_coords = np.concatenate([x.ravel(), y.ravel(), z.ravel()])
    axis_min, axis_max = all_coords.min(), all_coords.max()
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.set_zlim([axis_min, axis_max])
    ax.set_box_aspect((1, 1, 1))

    # 5. Limit tick density and suppress overlapping origin labels.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))

    # Increase tick padding for the oblique viewing angle.
    ax.tick_params(axis='x', pad=6, labelsize=12)
    ax.tick_params(axis='y', pad=6, labelsize=12)
    ax.tick_params(axis='z', pad=6, labelsize=12)

    # Format axis ticks in scientific notation when appropriate.
    axis_formatter = ScalarFormatter(useMathText=True)
    axis_formatter.set_scientific(True)
    axis_formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(axis_formatter)
    ax.yaxis.set_major_formatter(axis_formatter)
    ax.zaxis.set_major_formatter(axis_formatter)

    # 6. Axis labels.
    # ax.set_xlabel('X', labelpad=12)
    # ax.set_ylabel('Y', labelpad=12)
    # ax.set_zlabel('Z', labelpad=12)

    # 7. Viewing angle and uncluttered background.
    ax.view_init(elev=25, azim=-125)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # Hide pane edges to keep the directional surface visually isolated.
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    plt.tight_layout()
    return plt

@jit(nopython=True)
def transform(itr, tmx):
    """
    Perform a rotational transformation on itr based on tmx to generate a new fourth-order tensor
    """
    # Ensure the input tensor has the correct shape (3, 3, 3, 3)
    if itr.ndim != 4 or itr.shape != (3, 3, 3, 3):
        raise ValueError("Input tensor must have a shape of (3, 3, 3, 3)")

    # Get the size and number of dimensions of the input tensor
    ne = itr.size
    nd = itr.ndim
    # If the size is 3, no transformation is needed
    if ne == 3:
        return itr

    # Flatten the input tensor
    itr_tmp = itr.reshape(ne)
    # Initialize the output tensor
    otr = np.zeros_like(itr_tmp)
    # Calculate the transformation using broadcasting
    # Iterate over each element of the tensor (the 'oe' index)
    cne = np.cumprod(3 * np.ones(nd)) / 3
    for oe in range(ne):
        ioe = ((np.floor(oe / cne)) % 3).astype(np.int32)
        for ie in range(ne):
            iie = ((np.floor(ie / cne)) % 3).astype(np.int32)
            pmx = 1.0
            for id1 in range(nd):
                pmx *= tmx[ioe[id1], iie[id1]]
            otr[oe] += pmx * itr_tmp[ie]

    # Reshape the output tensor back to (3, 3, 3, 3)
    otr = otr.reshape((3, 3, 3, 3))

    return otr

@jit(nopython=True)
def sph2cart(a, e, r):
    """
    Convert spherical coordinates to Cartesian coordinates
    """
    x = r * np.cos(e) * np.cos(a)
    y = r * np.cos(e) * np.sin(a)
    z = r * np.sin(e)
    return x, y, z

if __name__ == '__main__':
    # Input the homogenized elasticity tensor
    savefigname = 'temp_save.png'
    CH = np.array([
        [236.655566856440, 196.499729888012, -134.968626666683, 0.000000000014, -0.035381885456, -0.000000000008],
        [196.499696960583, 236.662566801952, -134.971760374713, 0.000000000011, -0.033506102139, -0.000000000009],
        [-134.968562341910, -134.971698645201, 249.273253920426, -0.000000000500, 0.032649578937, 0.000000000005],
        [0.000000000005, 0.000000000010, -0.000000000011, 34.251899645954, -0.000000000295, 0.028851647477],
        [-0.035397419040, -0.033586847060, 0.032630190791, 0.000000000242, 34.251700714479, 0.000000000001],
        [-0.000000000095, -0.000000000060, 0.000000000017, 0.028805452978, -0.000000000138, 194.561448096773]
    ])

    plt = visual(CH)
    plt.savefig(savefigname)
    plt.show()
