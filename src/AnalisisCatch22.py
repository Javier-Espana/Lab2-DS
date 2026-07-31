import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import (dendrogram, fcluster, linkage,
                                     set_link_color_palette)
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_ROOT, 'results')

SUPERFICIE = '#fcfcfb'
TINTA = '#0b0b0b'
TINTA_SUAVE = '#52514e'
TINTA_TENUE = '#898781'
REJILLA = '#e1e0d9'
EJE = '#c3c2b7'
CATEGORICO = ['#2a78d6', '#eb6834', '#1baf7a']

SECUENCIAL = LinearSegmentedColormap.from_list(
    'SecuencialAzul',
    ['#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b'])

DIVERGENTE = LinearSegmentedColormap.from_list(
    'DivergenteAzulRojo',
    ['#0d366b', '#1c5cab', '#2a78d6', '#86b6ef', '#cde2fb',
     '#f0efec',
     '#fbdcdc', '#f0a3a3', '#e34948', '#b02b2b', '#6b1414'])

ETIQUETAS = {
    'Total_Consistent': 'Total',
    'Via_Aerea': 'Aerea',
    'Via_Terrestre': 'Terrestre',
    'Via_Maritima': 'Maritima',
    'Pais_El Salvador': 'El Salvador',
    'Pais_Guatemala': 'Guatemala',
    'Pais_Estados Unidos de América': 'Estados Unidos',
}


def Figura(ancho, alto):
    fig, ax = plt.subplots(figsize=(ancho, alto), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    for lado in ax.spines.values():
        lado.set_color(EJE)
        lado.set_linewidth(0.8)
    ax.tick_params(colors=TINTA_TENUE, labelsize=8)
    return fig, ax


def Guardar(fig, nombre):
    ruta = os.path.join(_RESULTS_DIR, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


def CargarMatriz():
    ruta = os.path.join(_RESULTS_DIR, 'Catch22Features.csv')
    return pd.read_csv(ruta, index_col='Serie')


def Estandarizar(matriz):
    valores = StandardScaler().fit_transform(matriz.values)
    return pd.DataFrame(valores, index=matriz.index, columns=matriz.columns)


def Agrupar(estandarizada):
    enlaces = linkage(estandarizada.values, method='ward')
    puntajes = {}
    for k in (2, 3, 4):
        etiquetas = fcluster(enlaces, k, criterion='maxclust')
        puntajes[k] = silhouette_score(estandarizada.values, etiquetas)
    mejor = max(puntajes, key=puntajes.get)
    grupos = fcluster(enlaces, mejor, criterion='maxclust')
    return enlaces, grupos, mejor, puntajes


def GraficarDendrograma(enlaces, estandarizada, k, grupos):
    set_link_color_palette(list(CATEGORICO))
    fig, ax = Figura(9.5, 5.5)
    dendrogram(enlaces, labels=[ETIQUETAS[s] for s in estandarizada.index],
               ax=ax, color_threshold=enlaces[-(k - 1), 2],
               above_threshold_color=TINTA_TENUE)

    grupo_de = {ETIQUETAS[s]: g for s, g in zip(estandarizada.index, grupos)}
    for etiqueta in ax.get_xticklabels():
        etiqueta.set_rotation(25)
        etiqueta.set_ha('right')
        etiqueta.set_fontsize(9)
        etiqueta.set_color(CATEGORICO[(grupo_de[etiqueta.get_text()] - 1) % len(CATEGORICO)])

    ax.set_title('Clustering jerarquico de las series (Ward sobre catch22)',
                 color=TINTA, fontsize=12, pad=12)
    ax.set_ylabel('Distancia de union', color=TINTA_SUAVE, fontsize=9)
    ax.grid(axis='y', color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)

    marcas = [Line2D([0], [0], marker='o', linestyle='', markersize=8,
                     color=CATEGORICO[(g - 1) % len(CATEGORICO)], label=f'Grupo {g}')
              for g in sorted(set(grupos))]
    leyenda = ax.legend(handles=marcas, frameon=False, fontsize=9, loc='upper right')
    for texto in leyenda.get_texts():
        texto.set_color(TINTA_SUAVE)
    return Guardar(fig, 'Catch22Dendrograma.png')


def GraficarPca(componentes, varianza, grupos, indice):
    fig, ax = Figura(8, 6.5)
    for g in sorted(set(grupos)):
        mascara = grupos == g
        ax.scatter(componentes[mascara, 0], componentes[mascara, 1],
                   s=110, color=CATEGORICO[(g - 1) % len(CATEGORICO)],
                   edgecolors=SUPERFICIE, linewidths=2, label=f'Grupo {g}', zorder=3)
    for i, serie in enumerate(indice):
        ax.annotate(ETIQUETAS[serie], (componentes[i, 0], componentes[i, 1]),
                    textcoords='offset points', xytext=(9, 5),
                    fontsize=9, color=TINTA_SUAVE)
    ax.axhline(0, color=EJE, linewidth=0.8, zorder=1)
    ax.axvline(0, color=EJE, linewidth=0.8, zorder=1)
    ax.grid(color=REJILLA, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(f'PC1 ({varianza[0] * 100:.1f}% de la varianza)',
                  color=TINTA_SUAVE, fontsize=10)
    ax.set_ylabel(f'PC2 ({varianza[1] * 100:.1f}% de la varianza)',
                  color=TINTA_SUAVE, fontsize=10)
    ax.set_title('PCA de las series en el espacio de catch22',
                 color=TINTA, fontsize=12, pad=12)
    leyenda = ax.legend(frameon=False, fontsize=9, loc='best')
    for texto in leyenda.get_texts():
        texto.set_color(TINTA_SUAVE)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    return Guardar(fig, 'Catch22Pca.png')


def GraficarHeatmap(estandarizada):
    fig, ax = Figura(15, 4.5)
    limite = float(np.abs(estandarizada.values).max())
    imagen = ax.imshow(estandarizada.values, cmap=DIVERGENTE,
                       vmin=-limite, vmax=limite, aspect='auto')
    ax.set_xticks(range(estandarizada.shape[1]))
    ax.set_xticklabels(estandarizada.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(estandarizada.shape[0]))
    ax.set_yticklabels([ETIQUETAS[s] for s in estandarizada.index], fontsize=9)
    ax.set_title('Caracteristicas de catch22 estandarizadas por serie',
                 color=TINTA, fontsize=12, pad=12)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.02, pad=0.01)
    barra.set_label('Desviaciones estandar', color=TINTA_SUAVE, fontsize=9)
    barra.ax.tick_params(colors=TINTA_TENUE, labelsize=8)
    barra.outline.set_edgecolor(EJE)
    return Guardar(fig, 'Catch22Heatmap.png')


def GraficarCorrelaciones(correlaciones):
    fig, ax = Figura(11, 9.5)
    imagen = ax.imshow(correlaciones.values, cmap=DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlaciones)))
    ax.set_xticklabels(correlaciones.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(correlaciones)))
    ax.set_yticklabels(correlaciones.index, fontsize=6.5)
    ax.set_title('Correlacion entre caracteristicas de catch22',
                 color=TINTA, fontsize=12, pad=12)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.02)
    barra.set_label('Correlacion de Pearson', color=TINTA_SUAVE, fontsize=9)
    barra.ax.tick_params(colors=TINTA_TENUE, labelsize=8)
    barra.outline.set_edgecolor(EJE)
    return Guardar(fig, 'Catch22Correlaciones.png')


def GraficarDistancias(distancias):
    fig, ax = Figura(7.5, 6.5)
    imagen = ax.imshow(distancias.values, cmap=SECUENCIAL, vmin=0)
    etiquetas = [ETIQUETAS[s] for s in distancias.index]
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas, fontsize=8)
    limite = distancias.values.max()
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            valor = distancias.values[i, j]
            ax.text(j, i, f'{valor:.1f}', ha='center', va='center', fontsize=8,
                    color='#ffffff' if valor > limite * 0.55 else TINTA)
    ax.set_title('Distancia euclidiana entre series (catch22 estandarizado)',
                 color=TINTA, fontsize=12, pad=12)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.02)
    barra.set_label('Distancia', color=TINTA_SUAVE, fontsize=9)
    barra.ax.tick_params(colors=TINTA_TENUE, labelsize=8)
    barra.outline.set_edgecolor(EJE)
    return Guardar(fig, 'Catch22Distancias.png')


def main():
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    matriz = CargarMatriz()
    print(f'2.3 Matriz de caracteristicas: {matriz.shape[0]} series x '
          f'{matriz.shape[1]} caracteristicas')

    estandarizada = Estandarizar(matriz)
    estandarizada.to_csv(os.path.join(_RESULTS_DIR, 'Catch22Estandarizado.csv'),
                         index_label='Serie')
    print(f'2.4 Estandarizada: media {estandarizada.values.mean():.2e}, '
          f'desviacion {estandarizada.values.std():.4f}')

    enlaces, grupos, k, puntajes = Agrupar(estandarizada)
    print(f'\n2.5 Clustering: silueta {[(n, round(v, 4)) for n, v in puntajes.items()]}, '
          f'k elegido = {k}')
    asignacion = pd.DataFrame({'Serie': estandarizada.index, 'Grupo': grupos})
    asignacion.to_csv(os.path.join(_RESULTS_DIR, 'Catch22Grupos.csv'), index=False)
    print(asignacion.to_string(index=False))
    GraficarDendrograma(enlaces, estandarizada, k, grupos)

    pca = PCA()
    componentes = pca.fit_transform(estandarizada.values)
    varianza = pca.explained_variance_ratio_
    pd.DataFrame({
        'Componente': [f'PC{i + 1}' for i in range(len(varianza))],
        'VarianzaExplicada': varianza.round(4),
        'VarianzaAcumulada': varianza.cumsum().round(4),
    }).to_csv(os.path.join(_RESULTS_DIR, 'Catch22PcaVarianza.csv'), index=False)
    cargas = pd.DataFrame(pca.components_[:2].T, index=matriz.columns,
                          columns=['PC1', 'PC2'])
    cargas.to_csv(os.path.join(_RESULTS_DIR, 'Catch22PcaCargas.csv'),
                  index_label='Caracteristica')
    GraficarPca(componentes, varianza, grupos, estandarizada.index)
    print(f'\nPCA: PC1 {varianza[0]:.1%}, PC2 {varianza[1]:.1%}, '
          f'acumulado {varianza[:2].sum():.1%}')
    print('\nCargas mas altas en PC1')
    print(cargas['PC1'].abs().sort_values(ascending=False).head(5).round(4).to_string())
    print('\nCargas mas altas en PC2')
    print(cargas['PC2'].abs().sort_values(ascending=False).head(5).round(4).to_string())

    GraficarHeatmap(estandarizada)

    correlaciones = estandarizada.corr()
    correlaciones.to_csv(os.path.join(_RESULTS_DIR, 'Catch22Correlaciones.csv'),
                         index_label='Caracteristica')
    GraficarCorrelaciones(correlaciones)

    distancias = pd.DataFrame(squareform(pdist(estandarizada.values)),
                              index=estandarizada.index, columns=estandarizada.index)
    distancias.round(4).to_csv(os.path.join(_RESULTS_DIR, 'Catch22Distancias.csv'),
                               index_label='Serie')
    GraficarDistancias(distancias)
    print('\nDistancias entre series')
    print(distancias.round(2).to_string())


if __name__ == '__main__':
    main()
