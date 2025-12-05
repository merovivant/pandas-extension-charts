import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

from wordcloud import WordCloud
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Decorador do pandas para criar um novo método para o DataFrame
@pd.api.extensions.register_dataframe_accessor("charts")
class ChartsAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if obj.empty:
            raise AttributeError("O DataFrame está vazio.")

    def waterfall(self, 
                   value_col, 
                   label_col, 
                   calc_total=True, 
                   total_label="Resultado Líquido",
                   figsize=(12, 8)):
        """
        Plota um gráfico Waterfall nativo.
        
        Parâmetros:
        - value_col: Nome da coluna com os valores (+/-)
        - label_col: Nome da coluna com as categorias
        - calc_total: Se True, adiciona automaticamente uma barra final com a soma.
        """
        df = self._obj.copy()
        
        if calc_total:
            total_val = df[value_col].sum()
            row_total = pd.DataFrame({label_col: [total_label], value_col: [total_val]})
            row_total['is_total'] = True 
            df['is_total'] = False
            df = pd.concat([df, row_total], ignore_index=True)
        else:
            df['is_total'] = False

        
        df['cumsum'] = df[value_col].cumsum()
        
        df['prev_cumsum'] = df['cumsum'].shift(1).fillna(0)
        
        df.loc[df['is_total'], 'prev_cumsum'] = 0

        
        df['plot_bottom'] = np.where(
            df[value_col] >= 0,
            df['prev_cumsum'],
            df['cumsum']
        )
        df.loc[df['is_total'], 'plot_bottom'] = 0

        df['plot_height'] = df[value_col].abs()

        def get_color(row):
            if row['is_total']: return '#1f77b4' # Azul padrão
            return '#2ca02c' if row[value_col] >= 0 else '#d62728' # Verde ou Vermelho

        colors = df.apply(get_color, axis=1)

        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(
            df.index, 
            df['plot_height'], 
            bottom=df['plot_bottom'], 
            color=colors,
            edgecolor='white',
            width=0.6,
            zorder=3
        )

        for i in range(len(df) - 1):
            y_start = df.loc[i, 'cumsum']
            
            ax.plot([i, i+1], [y_start, y_start], color='gray', linestyle='--', linewidth=1, zorder=1)

        for bar, value, cumsum in zip(bars, df[value_col], df['cumsum']):
            y_pos = bar.get_y() + bar.get_height()
            
            visual_top = bar.get_y() + bar.get_height()
            
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                visual_top + (max(df['cumsum']) * 0.02), # Offset dinâmico de 2%
                f"{value:+}", # Formata com sinal (+/-)
                ha='center', 
                va='bottom', 
                fontweight='bold',
                fontsize=10
            )

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df[label_col], rotation=0)
        
        ax.axhline(0, color='black', linewidth=1, zorder=2)
        
        ax.set_title(f"Fluxo de Valor: {label_col}", fontsize=14)
        ax.set_ylabel("Valor Monetário")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        return ax

    def gantt(self, 
               start_col, 
               label_col, 
               end_col=None, 
               duration_col=None, 
               color_col=None, 
               error_col=None,
               figsize=(16, 10)):
        """
        Plota um gráfico de Gantt nativo a partir do DataFrame.
        
        Parâmetros:
        - start_col: (str) Nome da coluna de início (datetime, string de data ou numérico).
        - label_col: (str) Coluna usada para o Eixo Y (Nomes das Tarefas).
        - end_col: (str, opcional) Nome da coluna de fim (datetime, string de data ou numérico).
        - duration_col: (str, opcional) Nome da coluna de duração.
        *Nota: Você deve fornecer 'end_col' OU 'duration_col'.
        *Nota: Strings de data (ex: '2023-11-01') são convertidas automaticamente para datetime.
        - color_col: (str, opcional) Coluna para definir as cores das barras.
        - error_col: (str, opcional) Coluna booleana. Se True, aplica hachura de erro.
        - figsize: (tuple) Tamanho da figura.
        """
        
        df = self._obj.copy()

        if end_col is None and duration_col is None:
            raise ValueError("Você precisa fornecer 'end_col' OU 'duration_col'")

        if df[start_col].dtype == 'object':
            df[start_col] = pd.to_datetime(df[start_col])
        
        if end_col and df[end_col].dtype == 'object':
            df[end_col] = pd.to_datetime(df[end_col])

        is_datetime = pd.api.types.is_datetime64_any_dtype(df[start_col])

        if is_datetime:
            df['start_num'] = mdates.date2num(df[start_col])
        else:
            df['start_num'] = df[start_col]

        if end_col:
            if is_datetime:
                df['end_num'] = mdates.date2num(df[end_col])
                df['duration_internal'] = df['end_num'] - df['start_num']
            else:
                df['duration_internal'] = df[end_col] - df[start_col]
        elif duration_col:
            df['duration_internal'] = df[duration_col]
            df['end_num'] = df['start_num'] + df['duration_internal']

        unique_labels = df[label_col].unique()
        y_map = {label: i for i, label in enumerate(unique_labels)}
        df['y_pos'] = df[label_col].map(y_map)

        if color_col is None:
            color_col = label_col
            
        unique_vals_color = df[color_col].unique()
        
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        safe_colors = [c for c in default_colors if c not in ['#d62728', 'red', 'r', '#ff0000', 'darkred']]
        
        color_map = {val: safe_colors[i % len(safe_colors)] for i, val in enumerate(unique_vals_color)}

        fig, ax = plt.subplots(figsize=figsize)

        for row in df.itertuples():
            c_val = getattr(row, color_col)
            color = color_map.get(c_val, 'grey')
            
            has_error = False
            if error_col and getattr(row, error_col):
                has_error = True
                
            hatch = '///' if has_error else None
            edgecolor = 'darkred' if has_error else 'white'
            
            ax.barh(
                row.y_pos, 
                row.duration_internal, 
                left=row.start_num, 
                height=0.6, 
                color=color,
                alpha=0.85,
                hatch=hatch,
                edgecolor=edgecolor,
                zorder=2
            )

            if is_datetime:
                dur_seconds = row.duration_internal * 24 * 3600
                if dur_seconds < 60:
                    dur_text = f"{dur_seconds:.1f}s"
                elif dur_seconds < 3600:
                    dur_text = f"{dur_seconds/60:.0f}min"
                else:
                    dur_text = f"{dur_seconds/3600:.1f}h"
            else:
                dur_text = f"{row.duration_internal:.2f}"
            
            if has_error:
                dur_text = f"{dur_text} (Erro)"

            text_x = row.start_num + (row.duration_internal / 2)
            
            ax.text(text_x, row.y_pos, dur_text, ha='center', va='center', 
                    color='white', fontsize=9, fontweight='bold', zorder=3)

        ax.set_yticks(range(len(unique_labels)))
        ax.set_yticklabels(unique_labels)
        ax.invert_yaxis()
        
        if is_datetime:
            ax.xaxis_date()
            fig.autofmt_xdate() 
        
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_title(f'Cronograma: {label_col}')
        ax.set_xlabel('Linha do Tempo')

        legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[k]) for k in color_map]
        legend_labels = list(color_map.keys())

        if error_col:
            error_patch = plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='darkred', hatch='///')
            
            legend_handles.append(error_patch)
            legend_labels.append("Com Erro")

        ax.legend(legend_handles, legend_labels, title=color_col, loc='upper right', bbox_to_anchor=(1.15, 1))


        plt.tight_layout()
        return ax

    def candlestick(self, 
                     date_col, 
                     open_col, 
                     high_col, 
                     low_col, 
                     close_col, 
                     figsize=(12, 6)):
        """
        Plota um gráfico de Candlestick (Velas) nativo.
        """
        df = self._obj.copy()
        
        if df[date_col].dtype == 'object':
            df[date_col] = pd.to_datetime(df[date_col])
        
        df['date_num'] = mdates.date2num(df[date_col])

        fig, ax = plt.subplots(figsize=figsize)

        width = 0.6
        width_wick = 0.05
        
        color_up = '#2ca02c'
        color_down = '#d62728'
        
        for row in df.itertuples():
            open_val = getattr(row, open_col)
            close_val = getattr(row, close_col)
            high_val = getattr(row, high_col)
            low_val = getattr(row, low_col)
            date_val = getattr(row, 'date_num')
            
            if close_val >= open_val:
                color = color_up
                bottom = open_val
                height = close_val - open_val
            else:
                color = color_down
                bottom = close_val
                height = open_val - close_val
                
            if height == 0:
                height = 0.01
                
            ax.plot([date_val, date_val], [low_val, high_val], color=color, linewidth=1.5, zorder=1)
            
            rect = Rectangle(
                xy=(date_val - width/2, bottom),
                width=width,
                height=height,
                facecolor=color,
                edgecolor=color,
                zorder=2
            )
            ax.add_patch(rect)

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        ax.set_title(f"Ação de Preço (OHLC)", fontsize=14)
        ax.set_ylabel("Preço")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        pad = 1
        ax.set_xlim(df['date_num'].min() - pad, df['date_num'].max() + pad)
        
        plt.tight_layout()
        return ax

    def radar(self, 
               categories_col, 
               value_cols, 
               title,
               max_val=10,
        ):
        """
        Plota um gráfico de Radar (Spider Chart) comparando múltiplas colunas (disciplinas)
        para cada linha (aluno).
        
        Parâmetros:
        - categories_col: Coluna com o nome das entidades (ex: Nome do Aluno).
        - value_cols: Lista de strings com os nomes das colunas de valores (ex: ['Mat', 'Port', ...]).
        - max_val: Valor máximo da escala (ex: 10 para notas, 100 para stats).
        """
        df = self._obj.copy()
        
        categories = value_cols
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], categories)
        
        ax.set_rlabel_position(0)
        plt.yticks(list(np.arange(0, max_val + 1, max_val/5)), color="grey", size=7)
        plt.ylim(0, max_val)
        
        colors = plt.cm.get_cmap("tab10", len(df))

        for index, row in df.iterrows():
            values = row[value_cols].tolist()
            
            values += values[:1]
            
            label = row[categories_col]
            color = colors(index)
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=color)
            
            ax.fill(angles, values, color=color, alpha=0.15)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
        plt.title(title, y=1.08, fontsize=16, fontweight='bold')
        
        return ax

    def wordcloud(self, 
                   text_col, 
                   title="Nuvem de Palavras", 
                   extra_stopwords=None,
                   figsize=(12, 8),
                   background_color='white'):
        """
        Gera e plota uma Nuvem de Palavras (WordCloud) a partir de uma coluna de texto.
        
        Parâmetros:
        - text_col: Nome da coluna que contém o texto.
        - title: Título do gráfico.
        - extra_stopwords: Lista de palavras extras para remover (ex: nome da empresa).
        - figsize: Tamanho da figura.
        - background_color: Cor de fundo da imagem.
        """
        df = self._obj.copy()
        
        text_data = df[text_col].dropna().astype(str)
        
        full_text = " ".join(text_data)
        
        stop_words_pt = set(stopwords.words('portuguese'))
        
        if extra_stopwords:
            stop_words_pt.update(extra_stopwords)
        
        stop_words_pt.update(['pra', 'tá', 'muito', 'fui', 'ser', 'vai'])

        wc = WordCloud(
            width=1600, 
            height=800,
            background_color=background_color,
            stopwords=stop_words_pt,
            min_font_size=10,
            colormap='viridis'
        ).generate(full_text)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation="bilinear")
        
        ax.axis("off")
        ax.set_title(title, fontsize=16, pad=20)
        
        plt.tight_layout()
        return ax