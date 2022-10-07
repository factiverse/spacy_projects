"""Class to retrieve information from Wikidata."""
from typing import Optional, Text

from wikidata.client import Client  # type: ignore
from wikidata.entity import Entity, EntityId  # type: ignore


class Wikidata:
    def __init__(self) -> None:
        """Retrieves Wikidata entities."""
        self._client = Client()
        self._wd_lang_code = {"en": "en", "no": "nb"}
        self._image_prop = self._client.get(EntityId("P18"))
        self._snl_prop = self._client.get(EntityId("P4342"))

    def get_wikidata_entity(self, wd_id: Optional[str]) -> Optional[Entity]:
        """Given a Wikidata id return the Wikidata entity object.

        Args:
            wd_id: Wikidata id in Q12345 format.

        Returns:
            Wikidata entity object.
        """
        return self._client.get(EntityId(wd_id), load=True)

    def get_image_url(self, entity: Entity) -> Text:
        """Gets image URL from the Wikidata entity.

        Args:
            entity: Wikidata entity to extract image from.

        Returns:
            Image URL if it exists othewise None.
        """
        image = entity.get(self._image_prop, None)
        return image.image_url if image else None

    def get_snl_id(
        self,
        wd_id: str,
    ) -> Optional[str]:
        """Given an entity spot retrieve the SNL id.

        Args:
            spot: Entity spot.

        Returns:
            SNL id from Wikidata if it exists otherwise None.
        """
        entity = self.get_wikidata_entity(wd_id)
        return (
            f"https://snl.no/{str(entity[self._snl_prop])}"
            if entity and self._snl_prop in entity
            else None
        )

    def get_wiki_url(
        self, entity: Optional[Entity], lang: str = "en"
    ) -> Optional[str]:  # noqa
        """Returns Wikipedia URL if it exists.

        Args:
            entity: Entity for which Wikipedia URL is needed.
            lang (optional): Language of the Wikipedia. Defaults to "en".

        Returns:
            Wikipedia URL if it exists None otherwise.
        """
        return (
            entity.attributes.get("sitelinks", {})  # type: ignore
            .get(f"{lang}wiki", {})
            .get("url", None)
            if entity
            else None
        )

    def get_wiki_title(self, wd_id: str, lang: str = "en") -> Optional[str]:
        """Returns Wikipedia title for the given Wikidata id.

        Args:
            wd_id: Wikidata id in Q12345 format.
            lang (optional): Language of the Wikipedia title. Defaults to "en".

        Returns:
            Wikipedia title if exists otherwise None.
        """
        if self.is_valid_id(wd_id):
            entity = self.get_wikidata_entity(wd_id)  # type: ignore
            url = self.get_wiki_url(entity, lang)  # type: ignore
            return url.rsplit("/", 1)[-1] if url else None
        return None

    def get_wikidata_description(
        self, wd_entity: Entity, lang: str = "en"
    ) -> Optional[str]:
        """Returns Wikidata description in the given language if it exists.

        Args:
            wd_entity: Wikidata entity.
            lang (optional): Language of the description. Defaults to "en".

        Returns:
            Description text if it exists otherwise None.
        """
        return wd_entity.description.texts.get(  # type: ignore
            self._wd_lang_code.get(lang, lang), None
        )
