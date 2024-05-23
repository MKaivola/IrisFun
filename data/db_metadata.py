from typing import Sequence

from sqlalchemy import create_engine, text, select, update
from sqlalchemy import MetaData, Table, Column, Integer, Float, Row

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

class Base(DeclarativeBase):
    pass

class TrainData(Base):

    __tablename__ = 'vowel_train'

    row_name: Mapped[int] = mapped_column(primary_key=True)
    y: Mapped[int]
    x_1: Mapped[float]
    x_2: Mapped[float]
    x_3: Mapped[float]
    x_4: Mapped[float]
    x_5: Mapped[float]
    x_6: Mapped[float]
    x_7: Mapped[float]
    x_8: Mapped[float]
    x_9: Mapped[float]
    x_10: Mapped[float]


class TestData(Base):

    __tablename__ = 'vowel_test'

    row_name: Mapped[int] = mapped_column(primary_key=True)
    y: Mapped[int]
    x_1: Mapped[float]
    x_2: Mapped[float]
    x_3: Mapped[float]
    x_4: Mapped[float]
    x_5: Mapped[float]
    x_6: Mapped[float]
    x_7: Mapped[float]
    x_8: Mapped[float]
    x_9: Mapped[float]
    x_10: Mapped[float]


class VowelDataBase():

    def __init__(self, connection_url: str) -> None:

        self.engine = create_engine(connection_url)

        self.metadata_obj = MetaData()

        self.train_data_table = Table(
            "vowel_train",
            self.metadata_obj,
            Column("row_name", Integer, primary_key=True),
            Column("y", Integer),
            Column("x_1", Float),
            Column("x_2", Float),
            Column("x_3", Float),
            Column("x_4", Float),
            Column("x_5", Float),
            Column("x_6", Float),
            Column("x_7", Float),
            Column("x_8", Float),
            Column("x_9", Float),
            Column("x_10", Float),
        )

        self.test_data_table = Table(
            "vowel_test",
            self.metadata_obj,
            Column("row_name", Integer, primary_key=True),
            Column("y", Integer),
            Column("x_1", Float),
            Column("x_2", Float),
            Column("x_3", Float),
            Column("x_4", Float),
            Column("x_5", Float),
            Column("x_6", Float),
            Column("x_7", Float),
            Column("x_8", Float),
            Column("x_9", Float),
            Column("x_10", Float),
        )

        self.new_data_table = Table(
            'vowel_new',
            self.metadata_obj,
            Column("row_name", Integer, primary_key=True),
            Column("y_pred", Integer),
            Column("x_1", Float),
            Column("x_2", Float),
            Column("x_3", Float),
            Column("x_4", Float),
            Column("x_5", Float),
            Column("x_6", Float),
            Column("x_7", Float),
            Column("x_8", Float),
            Column("x_9", Float),
            Column("x_10", Float),
        )
    
    def execute_select(self, select_args: list, where_args: list = None) -> Sequence[Row]:
        """
        Executes a SELECT statement against a database, returning all fetched rows as a sequence.
        Optionally includes a WHERE clause in the SELECT statement.
        
        Arguments
        ---------
        select_args
            List of positional arguments to pass to select function
        where_args
            List of positional arguments to pass to where function

        """
        stmt = select(*select_args)
        if where_args is not None:
            stmt = stmt.where(*where_args)

        with self.engine.begin() as conn:
            result = conn.execute(stmt)

        return result.all()

    def execute_update(self, update_args: list, values_args: dict,
                       where_args: list, data: list[dict]) -> None:
        """
        Executes an UPDATE statement against a database

        Arguments
        ---------
        update_args
            List of positional arguments to pass to update function
        values_args
            Dictionary specifying column:value pairs to update
        where_args
            List of positional arguments to pass to where function
        data
            List of parameter dictionaries to bind to a given statement

        """

        stmt = (update(*update_args)
                .values(values_args)
                .where(*where_args))
        
        with self.engine.begin() as conn:
            conn.execute(stmt,
                         data)
            