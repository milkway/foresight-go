# foresight-go (português)

> Versão resumida — a documentação completa e sempre atual está no [README em inglês](README.md).

Previsão de séries temporais em Go que escolhe o modelo pelo que teria
funcionado. O `foresight` ajusta vários modelos à série, refaz o passado para
ver como cada um teria se saído (backtest por origem móvel), escolhe pelo erro
fora da amostra e informa intervalos tirados dos erros de fato observados. Usa
só a biblioteca padrão e todo resultado é determinístico.

É a edição em Go do crate Rust [foresight](https://github.com/milkway/foresight),
escrita em Go puro: sem cgo, sem nada para ligar.

**Site:** <https://milkway.github.io/foresight-go/>

## Instalação

```bash
go get github.com/milkway/foresight-go
```

## Uso básico

```go
import foresight "github.com/milkway/foresight-go"

// dados mensais cuja primeira observação é de março
y := foresight.Monthly(valores, 2)
relatorio, err := foresight.DefaultBacktest().Run(y, foresight.Defaults())
if err != nil {
	log.Fatal(err)
}
melhor := relatorio.Best()
for _, p := range melhor.Forecast {
	faixa, _ := p.Interval(0.80)
	fmt.Println(p.Horizon, p.Mean, faixa.Lower, faixa.Upper)
}

// total dos próximos seis meses, com faixa própria
semestre, _ := melhor.Cumulative(6)
```

## O que tem

- Modelos: média, ingênuo, tendência, sazonal ingênuo, Theta, Holt-Winters,
  regressão log-linear (com deflator opcional), ARIMA sazonal por máxima
  verossimilhança exata (com regressores), ARIMA automático, família ETS com
  escolha automática, Prophet (quebras de tendência, eventos e degraus), TBATS
  (várias sazonalidades, de período não inteiro inclusive) e Croston, SBA e TSB
  para demanda intermitente.
- Decomposição STL e MSTL, e qualquer modelo sobre a série dessazonalizada.
- Ensemble: média, mediana, pesos pelo inverso do erro ou pesos empilhados.
- Limpeza: preenchimento de falhas e troca de valores atípicos.
- Qualquer modelo na escala log ou Box-Cox.
- Backtest por origem móvel em todos os núcleos, com MAPE, MAE, RMSE, MASE e
  viés por horizonte, média dos melhores modelos e escolha pelo erro fora da
  amostra.
- Intervalos empíricos por horizonte e para totais acumulados.

Os resultados são conferidos com o crate Rust (mesmos números) e com o pacote
`forecast` do R em dados públicos.

## Autores

- André Leite ([ORCID](https://orcid.org/0000-0002-4718-9766))
- Hugo Vasconcelos ([ORCID](https://orcid.org/0000-0001-6249-0920))
- Raydonal Ospina ([ORCID](https://orcid.org/0000-0002-9884-9090))

## Licença

MIT. Veja [LICENSE](LICENSE).
