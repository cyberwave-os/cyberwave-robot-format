# Copyright [2025] Tomáš Macháček <tomasmachacekw@gmail.com>
# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core format conversion engine and orchestration classes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .schema import CommonSchema, Material
from .utils import detect_format, validate_schema

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Exception raised during parsing errors."""

    pass


class ValidationError(Exception):
    """Exception raised during validation errors."""

    pass


@dataclass
class ParseContext:
    """Context information for parsing operations with enhanced error tracking."""

    file_path: Path
    base_dir: Path
    materials: dict[str, Material] = field(default_factory=dict)
    meshes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, message: str, element: str | None = None) -> None:
        """Add warning message with optional element context."""
        if element:
            message = f"{element}: {message}"
        self.warnings.append(message)
        logger.warning(message)

    def add_error(self, message: str, element: str | None = None) -> None:
        """Add error message with optional element context."""
        if element:
            message = f"{element}: {message}"
        self.errors.append(message)
        logger.error(message)


class BaseParser(ABC):
    """Base class for format parsers."""

    @abstractmethod
    def parse(self, input_path: str | Path) -> CommonSchema:
        """Parse input file and return common schema representation."""

    @abstractmethod
    def can_parse(self, file_path: str | Path) -> bool:
        """Check if this parser can handle the given file."""


class BaseExporter(ABC):
    """Base class for format exporters."""

    @abstractmethod
    def export(self, schema: CommonSchema, output_path: str | Path) -> None:
        """Export common schema to target format."""

    def get_extension(self) -> str:
        """Return the file extension for this format."""
        raise NotImplementedError("Subclasses must implement get_extension()")


class ConversionEngine:
    """Core conversion engine that orchestrates format conversions.

    The engine maintains a registry of parsers and exporters for different
    formats and handles the conversion workflow between them via the common
    schema intermediate representation.

    Example:
        >>> engine = ConversionEngine()
        >>> engine.register_parser('urdf', URDFParser())
        >>> engine.register_exporter('sdf', SDFExporter())
        >>> engine.convert('robot.urdf', 'robot.sdf')
    """

    def __init__(self):
        self.parsers: dict[str, BaseParser] = {}
        self.exporters: dict[str, BaseExporter] = {}

    def register_parser(self, format_name: str, parser: BaseParser) -> None:
        """Register a parser for a specific format."""
        self.parsers[format_name.lower()] = parser
        logger.debug("Registered parser for format: %s.", format_name)

    def register_exporter(self, format_name: str, exporter: BaseExporter) -> None:
        """Register an exporter for a specific format."""
        self.exporters[format_name.lower()] = exporter
        logger.debug("Registered exporter for format: %s.", format_name)

    def get_parser(self, format_name: str) -> BaseParser | None:
        """Get parser for a specific format."""
        return self.parsers.get(format_name.lower())

    def get_exporter(self, format_name: str) -> BaseExporter | None:
        """Get exporter for a specific format."""
        return self.exporters.get(format_name.lower())

    def detect_format(self, file_path: str | Path) -> str | None:
        """Detect format of a file using registered parsers."""
        file_path = Path(file_path)

        # Try each parser to see if it can parse the file
        for format_name, parser in self.parsers.items():
            if parser.can_parse(file_path):
                return format_name

        # Fall back to utility function
        return detect_format(file_path)

    def get_supported_formats(self) -> dict[str, list[str]]:
        """Return supported formats for parsing and exporting."""
        return {"parsers": list(self.parsers.keys()), "exporters": list(self.exporters.keys())}

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        source_format: str | None = None,
        target_format: str | None = None,
        validation: bool = True,
    ) -> CommonSchema:
        """Convert between robot description formats.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            source_format: Source format (auto-detected if None)
            target_format: Target format (inferred from extension if None)
            validation: Whether to validate schema during conversion

        Returns:
            CommonSchema representation of the robot model.

        Raises:
            ValueError: If formats are unsupported or conversion fails.
            FileNotFoundError: If input file doesn't exist.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Auto-detect source format if not specified
        if source_format is None:
            source_format = self.detect_format(input_path)
            if source_format is None:
                raise ValueError(f"Cannot detect format for: {input_path}")

        # Infer target format from extension if not specified
        if target_format is None:
            target_format = output_path.suffix.lstrip(".").lower()

        # Get appropriate parser and exporter
        parser = self.parsers.get(source_format.lower())
        if parser is None:
            raise ValueError(f"No parser found for format: {source_format}")

        exporter = self.exporters.get(target_format.lower())
        if exporter is None:
            raise ValueError(f"No exporter found for format: {target_format}")

        # Parse input to common schema
        logger.info("Parsing %s file: %s.", source_format.upper(), input_path)
        schema = parser.parse(input_path)

        # Validate schema if requested
        if validation:
            logger.debug("Validating intermediate schema.")
            validate_schema(schema)

        # Export to target format
        logger.info("Exporting to %s: %s.", target_format.upper(), output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exporter.export(schema, output_path)

        logger.info("Conversion complete: %s -> %s.", input_path, output_path)
        return schema


class FormatConverter:
    """High-level interface for robot format conversions.

    This class provides a simplified API for common conversion tasks,
    with built-in parsers and exporters for standard formats.

    Example:
        >>> converter = FormatConverter()
        >>> converter.urdf_to_sdf('robot.urdf', 'robot.sdf')
        >>> converter.batch_convert('models/', 'output/', 'urdf', 'mjcf')
    """

    def __init__(self):
        self.engine = ConversionEngine()
        self._register_default_processors()

    def _register_default_processors(self) -> None:
        """Register default parsers and exporters for standard formats."""
        # Import from submodules
        from .mjcf import MJCFExporter, MJCFParser
        from .obj import OBJMTLExporter, OBJMTLParser
        from .schema_io import SchemaExporter, SchemaParser
        from .sdf import SDFExporter, SDFParser
        from .urdf import URDFExporter, URDFParser

        # Register parsers
        self.engine.register_parser("urdf", URDFParser())
        self.engine.register_parser("sdf", SDFParser())
        self.engine.register_parser("mjcf", MJCFParser())
        self.engine.register_parser("xml", MJCFParser())  # MJCF is XML
        self.engine.register_parser("schema", SchemaParser())
        self.engine.register_parser("yaml", SchemaParser())
        self.engine.register_parser("json", SchemaParser())
        self.engine.register_parser("obj", OBJMTLParser())

        # Register exporters
        self.engine.register_exporter("urdf", URDFExporter())
        self.engine.register_exporter("sdf", SDFExporter())
        self.engine.register_exporter("mjcf", MJCFExporter())
        self.engine.register_exporter("xml", MJCFExporter())
        self.engine.register_exporter("schema", SchemaExporter())
        self.engine.register_exporter("yaml", SchemaExporter())
        self.engine.register_exporter("json", SchemaExporter())
        self.engine.register_exporter("obj", OBJMTLExporter())

        # Try to register USD support if available
        try:
            from .usd import USDExporter, USDParser

            self.engine.register_parser("usd", USDParser())
            self.engine.register_parser("usda", USDParser())
            self.engine.register_exporter("usd", USDExporter())
            self.engine.register_exporter("usda", USDExporter())
        except ImportError:
            logger.debug("USD support not available (missing pxr module).")

    def convert(self, input_path: str | Path, output_path: str | Path, **kwargs) -> CommonSchema:
        """Convert between formats using the conversion engine."""
        return self.engine.convert(input_path, output_path, **kwargs)

    def urdf_to_sdf(self, urdf_path: str, sdf_path: str) -> CommonSchema:
        """Convert URDF to SDF format."""
        return self.convert(urdf_path, sdf_path, source_format="urdf", target_format="sdf")

    def urdf_to_mjcf(self, urdf_path: str, mjcf_path: str) -> CommonSchema:
        """Convert URDF to MJCF format."""
        return self.convert(urdf_path, mjcf_path, source_format="urdf", target_format="mjcf")

    def sdf_to_urdf(self, sdf_path: str, urdf_path: str) -> CommonSchema:
        """Convert SDF to URDF format."""
        return self.convert(sdf_path, urdf_path, source_format="sdf", target_format="urdf")

    def to_schema(self, input_path: str, schema_path: str) -> CommonSchema:
        """Convert any supported format to common schema."""
        return self.convert(input_path, schema_path, target_format="schema")

    def from_schema(self, schema_path: str, output_path: str, target_format: str | None = None) -> CommonSchema:
        """Convert from common schema to any supported format."""
        return self.convert(schema_path, output_path, source_format="schema", target_format=target_format)

    def batch_convert(
        self, input_dir: str | Path, output_dir: str | Path, source_format: str, target_format: str, pattern: str = "*"
    ) -> list[Path]:
        """Batch convert files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            source_format: Source file format
            target_format: Target file format
            pattern: File pattern to match (default: "*")

        Returns:
            List of successfully converted output files.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find matching files
        # If pattern already includes extension, use it directly
        # Otherwise, append the source format extension
        if pattern.endswith(f".{source_format}") or "*" in pattern:
            # Pattern already specifies the extension
            input_files = list(input_dir.glob(pattern))
        else:
            ext = f".{source_format}"
            input_files = list(input_dir.glob(f"{pattern}{ext}"))

        if not input_files:
            logger.warning("No %s files found in %s.", source_format, input_dir)
            return []

        converted_files = []
        target_ext = f".{target_format}"

        for input_file in input_files:
            output_file = output_dir / (input_file.stem + target_ext)
            try:
                self.convert(input_file, output_file, source_format=source_format, target_format=target_format)
                converted_files.append(output_file)
            except Exception as e:
                logger.error("Failed to convert %s: %s.", input_file, e)

        logger.info("Batch conversion complete: %s/%s files converted.", len(converted_files), len(input_files))
        return converted_files

    def get_conversion_matrix(self) -> dict[str, list[str]]:
        """Get matrix of supported conversion paths."""
        formats = self.engine.get_supported_formats()
        matrix = {}

        for source_fmt in formats["parsers"]:
            matrix[source_fmt] = formats["exporters"]

        return matrix
