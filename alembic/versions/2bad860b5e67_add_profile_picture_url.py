"""add_profile_picture_url

Revision ID: 2bad860b5e67
Revises: 416bf29a2aeb
Create Date: 2025-01-09 06:09:29.056245

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2bad860b5e67'
down_revision: Union[str, None] = '416bf29a2aeb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add profile_picture_url column
    op.add_column('users', sa.Column('profile_picture_url', sa.String(255), nullable=True))

def downgrade():
    # Remove profile_picture_url column
    op.drop_column('users', 'profile_picture_url')
