"""Demo of separating metric expressions from their execution contexts.

The module defines a tiny expression tree along with two compilers:

* ``SQLCompiler`` renders the tree into a SQL statement.
* ``PandasCompiler`` executes the same logical metric on an in-memory DataFrame.

Running this file prints both representations for an example metric plus a
filtered variant that demonstrates extensibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd


class Expression:
    """Base node for all expressions."""

    def accept(self, compiler: "ExpressionCompiler"):
        method_name = f"visit_{self.__class__.__name__}"
        method = getattr(compiler, method_name, None)
        if method is None:
            raise NotImplementedError(f"{compiler.__class__.__name__} cannot visit {self.__class__.__name__}")
        return method(self)


class Dataset(Expression):
    def __init__(self, name: str):
        self.name = name


class Feature(Expression):
    def __init__(self, name: str, dataset: Dataset | None = None):
        self.name = name
        self.dataset = dataset


class Literal(Expression):
    def __init__(self, value: Union[int, float, str]):
        self.value = value


class Average(Expression):
    def __init__(self, child: Expression, alias: str | None = None):
        self.child = child
        self.alias = alias


class GroupBy(Expression):
    def __init__(self, source: Expression, metric: Expression, by: Union[str, List[str]]):
        self.source = source
        self.metric = metric
        if isinstance(by, str):
            self.by = [by]
        else:
            self.by = by


class Filter(Expression):
    def __init__(self, source: Expression, feature: str, op: str, value: Literal):
        self.source = source
        self.feature = feature
        self.op = op
        self.value = value


@dataclass
class AggregationPlan:
    function: str
    column: str
    alias: str


class ExpressionCompiler:
    def compile(self, expr: Expression):
        return expr.accept(self)


class SQLCompiler(ExpressionCompiler):
    """Render the expression tree into a SQL statement."""

    def compile_Dataset(self, expr: Dataset) -> str:
        return expr.name

    def compile_Feature(self, expr: Feature) -> str:
        if expr.dataset:
            return f"{expr.dataset.name}.{expr.name}"
        return expr.name

    def compile_Literal(self, expr: Literal) -> str:
        if isinstance(expr.value, str):
            return f"'{expr.value}'"
        return str(expr.value)

    def compile_Average(self, expr: Average) -> AggregationPlan:
        feature_sql = self.compile(expr.child)
        alias = expr.alias or f"avg_{expr.child.name}"
        return AggregationPlan("AVG", feature_sql, alias)

    def compile_GroupBy(self, expr: GroupBy) -> str:
        source_sql = self.compile(expr.source)
        plan = self.compile(expr.metric)
        group_clause = ", ".join(expr.by)
        sql = [
            "SELECT",
            f"    {group_clause},",
            f"    {plan.function}({plan.column}) AS {plan.alias}",
            f"FROM {source_sql}",
        ]
        sql.append(f"GROUP BY {group_clause};")
        return "\n".join(sql)

    def compile_Filter(self, expr: Filter) -> str:
        source_sql = self.compile(expr.source)
        value_sql = self.compile(expr.value)
        condition = f"{expr.feature} {expr.op} {value_sql}"
        return f"(SELECT * FROM {source_sql} WHERE {condition})"


class PandasCompiler(ExpressionCompiler):
    """Execute the expression tree against in-memory DataFrames."""

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets

    def compile_Dataset(self, expr: Dataset) -> pd.DataFrame:
        return self.datasets[expr.name]

    def compile_Feature(self, expr: Feature) -> str:
        return expr.name

    def compile_Literal(self, expr: Literal):
        return expr.value

    def compile_Average(self, expr: Average) -> AggregationPlan:
        feature_name = self.compile(expr.child)
        alias = expr.alias or f"avg_{feature_name}"
        return AggregationPlan("AVG", feature_name, alias)

    def compile_GroupBy(self, expr: GroupBy) -> pd.DataFrame:
        df = self.compile(expr.source)
        plan = self.compile(expr.metric)
        grouped = df.groupby(expr.by)
        if plan.function == "AVG":
            series = grouped[plan.column].mean()
        else:
            raise ValueError(f"Unsupported aggregation: {plan.function}")
        return series.reset_index(name=plan.alias)

    def compile_Filter(self, expr: Filter) -> pd.DataFrame:
        df = self.compile(expr.source)
        value = self.compile(expr.value)
        feature = expr.feature
        if expr.op == ">":
            mask = df[feature] > value
        elif expr.op == ">=":
            mask = df[feature] >= value
        elif expr.op == "<":
            mask = df[feature] < value
        elif expr.op == "<=":
            mask = df[feature] <= value
        elif expr.op == "==":
            mask = df[feature] == value
        elif expr.op == "!=":
            mask = df[feature] != value
        else:
            raise ValueError(f"Unsupported operator: {expr.op}")
        return df.loc[mask]


def demo():
    sessions = Dataset("sessions")
    metric = Average(Feature("session_duration", dataset=sessions), alias="avg_duration")
    base_expr = GroupBy(source=sessions, metric=metric, by="region")

    filtered_expr = GroupBy(
        source=Filter(sessions, feature="session_duration", op=">", value=Literal(10)),
        metric=metric,
        by="region",
    )

    df = pd.DataFrame(
        [
            {"user": "u1", "region": "NA", "session_duration": 30},
            {"user": "u2", "region": "EU", "session_duration": 25},
            {"user": "u3", "region": "NA", "session_duration": 45},
            {"user": "u4", "region": "APAC", "session_duration": 10},
        ]
    )

    datasets = {"sessions": df}

    sql_compiler = SQLCompiler()
    pandas_compiler = PandasCompiler(datasets)

    print(" Base Metric as SQL ")
    print(sql_compiler.compile(base_expr))

    print("\n Base Metric via Pandas ")
    print(pandas_compiler.compile(base_expr))

    print("\n Filtered Metric as SQL ")
    print(sql_compiler.compile(filtered_expr))

    print("\n Filtered Metric via Pandas ")
    print(pandas_compiler.compile(filtered_expr))


if __name__ == "__main__":
    demo()
