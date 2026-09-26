class PrimaryRouter:
    """
    A router to control all database operations on models in the
    various applications. Currently routes everything to the default
    PostGIS database. Can be extended for multi-DB support in the future.
    """
    def db_for_read(self, model, **hints):
        """
        All reads go to the default (PostGIS) database.
        """
        return 'default'

    def db_for_write(self, model, **hints):
        """
        All writes go to the default (PostGIS) database.
        """
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if both models are in the same database.
        """
        if obj1._state.db == obj2._state.db:
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        All apps can migrate on the default database.
        """
        return db == 'default'
